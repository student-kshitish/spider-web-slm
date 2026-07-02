import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLookbackAttention(nn.Module):
    """
    Surgical lookback attention, gated by a learned per-token
    "needs-reference-resolution" gate. This is the attention half of the
    unified hybrid: the Spider Web rings keep doing routing/computation for
    EVERY token; this module adds content-based causal attention ONLY for the
    tokens the gate flags, and fuses the result back gate-weighted.

    Why a NEW gate might learn where the storage gate did not
    --------------------------------------------------------
    The separable-write STORAGE gate decided, at WRITE time u, "is this an
    entity worth storing?". Its payoff only appeared much later, at READ time
    t >> u, when some downstream token finally needed that entity. The
    supervising signal (lower CE) was DISTANT from the decision, so the gate
    received essentially no usable gradient and collapsed to ~0.

    This LOOKBACK gate decides, at READ time t, "does token t need to look
    back?". Its payoff — a better prediction of token t+1 — is IMMEDIATE and at
    the SAME position t. CE_t depends directly on gate_t, so the gradient on the
    gate is local. The whole point of Step 2 is to test whether that immediacy
    actually makes the gate learnable.

    Mechanism
    ---------
        gate_t = sigmoid(MLP(x_t))                        in (0,1)
        causal content attention of x_t over prior states x_{u<=t}:
            bounded width N : u in [t-N+1, t]   (last N tokens, incl. self)
            full   width    : u in [0, t]
            attn_t = softmax_u( q_t . k_u / sqrt(d) )      over allowed u
            retrieved_t = sum_u attn_{t,u} v_u
        x_out_t = x_t + gate_t * o_proj(retrieved_t)       # gate-weighted fuse

    Width is configurable per-call (`width`) or via config.model.lookback_width.
    width <= 0 (or None) means FULL lookback.

    Warm-start
    ----------
    o_proj is zero-initialised, so retrieved contributes nothing at init and
    x_out == x exactly: a warm restart preserves the baseline CE. The gate MLP
    output bias is zero, so the gate starts NEUTRAL (~0.5 on-rate) and is free
    to move EITHER way during training — it is deliberately NOT biased shut the
    way the storage gate was. Because o_proj starts at zero, the gate gets no
    gradient on step 0; it begins to learn only once o_proj moves off zero,
    which is exactly the immediate-signal pathway we are probing.
    """

    def __init__(self, config, lookback_width=32):
        super().__init__()
        d = config.model.dim
        self.d = d
        # None / <=0 -> full lookback (all prior positions)
        self.lookback_width = lookback_width

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        hg = max(16, d // 2)
        self.gate_mlp = nn.Sequential(           # learned "needs lookback" gate
            nn.Linear(d, hg), nn.GELU(), nn.Linear(hg, 1),
        )

        # identity init: zero o_proj -> warm-start CE == baseline.
        nn.init.zeros_(self.o_proj.weight)
        # neutral gate (bias 0 -> ~0.5 on-rate); free to rise or fall.
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, x, width="default"):
        """
        x : (B, T, d)
        width : "default" -> use self.lookback_width;  int N -> bounded last N;
                None or <=0 -> full lookback.
        returns (x_out (B,T,d), stats dict). stats["gate"] is the (B,T) gate
        tensor (kept on-graph) for per-token CE-by-flag diagnostics.
        """
        B, T, d = x.shape
        if width == "default":
            width = self.lookback_width

        gate = torch.sigmoid(self.gate_mlp(x)).squeeze(-1)            # (B,T)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.einsum("btd,bud->btu", q, k) / (d ** 0.5)      # (B,T,u)
        scores = scores.float()

        t_idx = torch.arange(T, device=x.device).view(T, 1)
        u_idx = torch.arange(T, device=x.device).view(1, T)
        allowed = u_idx <= t_idx                                      # causal
        bounded = (width is not None) and (width > 0)
        if bounded:
            allowed = allowed & ((t_idx - u_idx) <= (width - 1))      # last N

        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(q.dtype)             # (B,T,u)
        retrieved = torch.einsum("btu,bud->btd", attn, v)            # (B,T,d)

        x_out = x + gate.unsqueeze(-1) * self.o_proj(retrieved)

        with torch.no_grad():
            # off-self attention mass: how much a flagged token actually reaches
            # BACK rather than attending to itself (a real-lookback signal).
            self_mass = attn.float().diagonal(dim1=1, dim2=2)         # (B,T)
            stats = {
                "gate":            gate,                              # (B,T) on-graph
                "gate_mean":       float(gate.mean()),
                "gate_frac_on":    float((gate > 0.5).float().mean()),
                "lookback_frac":   float((1.0 - self_mass).mean()),
                "width":           (-1 if not bounded else int(width)),
                "o_proj_norm":     float(self.o_proj.weight.norm()),
            }
        # attention distribution (B,T,u), kept ON-GRAPH for the optional
        # retrieval-supervision aux (CE pulling attn[recall]->source). Added
        # after the no_grad block so it stays differentiable.
        stats["attn"] = attn
        return x_out, stats
