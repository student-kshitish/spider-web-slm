import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from core.lorenz import LorenzRouter
from core.memory import SolarRingMemory


class WebNode(nn.Module):
    """
    Final Stable WebNode for Spider Web SLM

    Components:
    - Lateral Attention (ring communication)
    - SwiGLU FFN (reasoning)
    - SRM Memory (context)
    - Lorenz Router (dynamic routing)
    - Aux Projection (light supervision)
    """

    def __init__(self, config):
        super().__init__()

        self.d = config.model.dim
        self.hidden_dim = config.model.hidden_dim

        # -------------------------
        # 1. LATERAL ATTENTION
        # -------------------------
        self.lateral_attn = nn.MultiheadAttention(
            self.d, num_heads=4, batch_first=True, dropout=0.0
        )
        self.attn_norm = nn.RMSNorm(self.d)

        # -------------------------
        # 2. SWIGLU FFN
        # -------------------------
        self.w1 = spectral_norm(nn.Linear(self.d, self.hidden_dim, bias=False))
        self.w2 = spectral_norm(nn.Linear(self.hidden_dim, self.d, bias=False))
        self.w3 = spectral_norm(nn.Linear(self.d, self.hidden_dim, bias=False))
        self.ffn_norm = nn.RMSNorm(self.d)

        # -------------------------
        # 3. MEMORY + ROUTER
        # -------------------------
        self.memory = SolarRingMemory(config)
        self.router = LorenzRouter(config)

        # -------------------------
        # 4. AUX PROJECTION (FIXED)
        # -------------------------
        self.aux_proj = spectral_norm(
            nn.Linear(self.d, self.d, bias=False)
        )

    # -------------------------
    # SWIGLU
    # -------------------------
    def swiglu(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(
        self,
        h: torch.Tensor,            # (N, d)
        neighbours: torch.Tensor,  # (N, K, d)
        m_t_prev: torch.Tensor,    # (N, slots, d)
        m_s_above: torch.Tensor = None,
        tau: float = 1.0,
        hard: bool = False,
    ) -> dict:

        # -------------------------
        # 1. MEMORY READ
        # -------------------------
        h_enriched, m_t_next = self.memory(h, m_t_prev, m_s_above)

        # -------------------------
        # 2. LATERAL ATTENTION
        # -------------------------
        q = h_enriched.unsqueeze(1)  # (N,1,d)

        if neighbours is not None and neighbours.numel() > 0:
            kv = torch.cat([q, neighbours], dim=1)
        else:
            kv = q

        attn_out, _ = self.lateral_attn(q, kv, kv)
        h = self.attn_norm(h_enriched + attn_out.squeeze(1))

        # -------------------------
        # 3. FFN (SWIGLU)
        # -------------------------
        h = self.ffn_norm(h + self.swiglu(h))

        # -------------------------
        # 4. ROUTING
        # -------------------------
        routing_out = self.router(h, m_t_next, tau=tau, hard=hard)

        # support both API styles
        if isinstance(routing_out, tuple):
            routing, h_guarded = routing_out
        else:
            routing = routing_out
            h_guarded = h

        # -------------------------
        # 5. AUX PROJECTION (FIXED)
        # -------------------------
        aux_logits = self.aux_proj(h_guarded)  # (N, d)

        # -------------------------
        # OUTPUT
        # -------------------------
        return {
            "h": h_guarded,
            "routing": routing,
            "m_t": m_t_next,
            "aux_logits": aux_logits,
        }