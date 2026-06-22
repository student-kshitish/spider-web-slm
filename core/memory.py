import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class SolarRingMemory(nn.Module):
    """
    Solar Ring Memory (SRM) v2.1 — The Gravitational Engine
    """

    def __init__(self, config):
        super().__init__()

        self.d = config.model.dim
        self.slots = config.memory.slots
        self.alpha = config.memory.alpha
        self.beta = config.memory.beta

        # -------------------------
        # INITIAL MEMORY
        # -------------------------
        self.m_t_seed = nn.Parameter(torch.randn(self.slots, self.d) * 0.02)

        # -------------------------
        # PROJECTIONS
        # -------------------------
        self.q_proj = spectral_norm(nn.Linear(self.d, self.d, bias=False))
        self.k_proj = spectral_norm(nn.Linear(self.d, self.d, bias=False))
        self.gate_proj = spectral_norm(nn.Linear(self.d, self.d, bias=False))
        self.spatial_compress = spectral_norm(nn.Linear(self.d, self.d, bias=False))

        # -------------------------
        # STABILITY
        # -------------------------
        self.norm = nn.RMSNorm(self.d)

    # -------------------------
    # INIT MEMORY (FIXED)
    # -------------------------
    def init_memory(self, batch_size: int, device: torch.device):
        m_t = self.m_t_seed.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)

        # spatial memory (empty start)
        m_s = torch.zeros_like(m_t)

        return m_t, m_s

    # -------------------------
    # GRAVITY
    # -------------------------
    def gravity_read(self, h, M):
        q = self.q_proj(h).unsqueeze(1)        # (B,1,d)
        k = self.k_proj(M)                     # (B,slots,d)

        scores = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        # Softmax in float32 for numerical stability; cast back so downstream
        # multiplications (payload, m_t_next) stay in the model's working dtype.
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)

        context = (weights @ M).squeeze(1)

        return context, weights.squeeze(1)

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, h, m_t_prev, m_s_above=None):

        # --- Spatial ---
        if m_s_above is not None:
            m_s_local = torch.tanh(self.spatial_compress(m_s_above))
            ctx_s, _ = self.gravity_read(h, m_s_local)
            h = h + 0.1 * ctx_s

        # --- Temporal ---
        ctx_t, g_weights = self.gravity_read(h, m_t_prev)

        # --- Gate ---
        gate = torch.sigmoid(self.gate_proj(h))

        payload = gate * h

        m_t_next = self.alpha * m_t_prev + self.beta * payload.unsqueeze(1)

        # stability
        m_t_next = self.norm(m_t_next)

        # output
        h_out = h + 0.5 * ctx_t

        return h_out, m_t_next