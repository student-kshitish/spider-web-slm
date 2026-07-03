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


class NameLookback(nn.Module):
    """
    Learned NAME-lookback (Step 2 — removes the last oracle).

    The name_transport experiment proved that if the slot key at each object-intro
    / recall position is sourced from the governing NAME token's embedding, the
    learned router forms subject-keyed addresses and binds selectively (2AFC
    53%->82%). But it still HANDED the model the name's POSITION (name_src). This
    module removes that oracle: it LOCATES the governing name itself.

    Mechanism (a causal single-head attention)
    ------------------------------------------
        q_t = W_q x_t ;  k_u = W_k x_u          (from the MIXED hidden state x,
                                                 which carries the positional /
                                                 syntactic context needed to know
                                                 "which earlier name governs me")
        a[t,u] = softmax_{u<=t}( q_t . k_u / sqrt(d) )        # causal
        name_hat[t] = sum_{u<=t} a[t,u] * token_emb[u]        # embed(input_ids[u])

    CRITICAL — the VALUE is the RAW EMBEDDING token_emb = embed(input_ids), NOT the
    mixed state x. The name identity must ride VERBATIM into the key (exactly the
    mem_copy embedding-transport path). Transporting mixed x instead would smear
    the name the same way addr_learned failed. There is deliberately NO v_proj:
    the transported vector is the embedding itself, unmodified.

    Supervision (train only, gated by --w_name)
    -------------------------------------------
    The attention `a` is exposed on-graph (stats["attn"]) and pulled toward the
    known name positions via an NLL aux (train_attn_super.name_supervision). Those
    labels (name_src) are used ONLY in that loss — this forward NEVER receives
    them, so at inference the module must find the name unaided.

    Init: q_proj/k_proj get the default (small random) init, so scores are ~0 and
    the attention starts DIFFUSE (high entropy) — expected; the supervision sharpens
    it. (Zero-init BOTH would kill the q/k cross-gradient, so it is avoided.)
    """

    def __init__(self, config):
        super().__init__()
        d = config.model.dim
        self.d = d
        self.q_proj = nn.Linear(d, d, bias=False)   # query from mixed state x
        self.k_proj = nn.Linear(d, d, bias=False)   # key   from mixed state x
        # NO v_proj / o_proj: the value is the verbatim token embedding, and the
        # output (name_hat) is fed straight in as the addressing key.

    def forward(self, x, token_emb):
        """
        x         : (B,T,d)  mixed hidden state — QUERY/KEY source (context to locate).
        token_emb : (B,T,d)  raw embed(input_ids) — the VALUE, transported verbatim.
        returns (name_hat (B,T,d), stats). stats["attn"] (B,T,u) is on-graph for the
        name-supervision aux; name_hat is the identity-preserving lookback output.
        """
        B, T, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = torch.einsum("btd,bud->btu", q, k) / (d ** 0.5)     # (B,T,u)
        scores = scores.float()

        t_idx = torch.arange(T, device=x.device).view(T, 1)
        u_idx = torch.arange(T, device=x.device).view(1, T)
        allowed = u_idx <= t_idx                                     # causal (u <= t)
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)                        # (B,T,u) float

        # IDENTITY TRANSPORT: name_hat[t] = sum_u a[t,u] * embed(input_ids[u]).
        # token_emb is the raw embedding table lookup, so the name rides verbatim.
        name_hat = torch.einsum("btu,bud->btd", attn.to(token_emb.dtype), token_emb)

        with torch.no_grad():
            ent = -(attn.clamp_min(1e-9).log() * attn).sum(-1).mean()   # attention entropy
            stats = {
                "entropy":   float(ent),
                "self_mass": float(attn.diagonal(dim1=1, dim2=2).mean()),
            }
        stats["attn"] = attn        # (B,T,u) ON-GRAPH — name-supervision target
        return name_hat, stats
