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
        # write policy: "blend" (mix into all slots) or "separable" (one slot
        # per entity, round-robin, occupied slots protected). Default keeps the
        # original behaviour so trained checkpoints are unaffected.
        self.write_mode = getattr(config.memory, "write_mode", "blend")
        # per-batch round-robin write pointer for separable mode (lazy-init,
        # reset by init_memory). None in blend mode.
        self._write_ptr = None

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

        # reset the round-robin write pointer (separable mode starts at slot 0)
        self._write_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)

        return m_t, m_s

    # -------------------------
    # WRITE  (blend vs separable)
    # -------------------------
    def write(self, M, payload, write_ptr=None):
        """
        Update memory M with one new entity `payload`, per the write policy.

        M        : (B, slots, d)   current memory bank
        payload  : (B, d)          gated content for this step ("one entity")
        write_ptr: (B,) long | None  round-robin slot pointer (separable only).
                   If None, falls back to self._write_ptr (or zeros).

        returns (M_new, write_ptr_next)

        ---- "blend" (current) -------------------------------------------------
            M_new = alpha * M + beta * payload     (broadcast to EVERY slot)
        Every slot decays and every slot receives the new payload, so distinct
        entities written over time superpose into the same slots. write_ptr is
        passed through untouched.

        ---- "separable" -------------------------------------------------------
        The entity is written to ITS OWN slot, chosen round-robin by write_ptr.
        Only that one slot is overwritten; all other (occupied) slots are left
        exactly as they were — no alpha decay, no beta blend. The pointer then
        advances (mod slots) so the next distinct entity lands in a fresh slot.
        """
        B, S, d = M.shape
        if self.write_mode == "blend":
            M_new = self.alpha * M + self.beta * payload.unsqueeze(1)
            return M_new, write_ptr

        if self.write_mode == "separable":
            if write_ptr is None:
                write_ptr = self._write_ptr
            if write_ptr is None:
                write_ptr = torch.zeros(B, dtype=torch.long, device=M.device)
            idx = write_ptr % S                                  # (B,)
            M_new = M.clone()                                    # protect occupied slots
            M_new[torch.arange(B, device=M.device), idx] = payload.to(M_new.dtype)
            return M_new, write_ptr + 1

        raise ValueError(f"unknown write_mode: {self.write_mode!r}")

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

        # NOTE: the hop-loop SRM is local WORKING memory (it is mean-pooled across
        # tokens in web.py), so it always blends — the separable/non-blending
        # ENTITY store is a separate post-hop module (SeparableMemoryRead), driven
        # by config.memory.write_mode. The write() method above is kept for the
        # standalone mechanism unit tests, not used on this path.
        m_t_next = self.alpha * m_t_prev + self.beta * payload.unsqueeze(1)

        # stability
        m_t_next = self.norm(m_t_next)

        # output
        h_out = h + 0.5 * ctx_t

        return h_out, m_t_next


class OrbitalQueryRead(nn.Module):
    """
    Orbital query read — content-addressable retrieval over ALL memory slots.

    Replaces the fixed-lag recall objective with a learned, query-driven read:

      1. From each position's hidden state h, project a QUERY q.
      2. Score q against every memory slot (pooled across all rings/nodes)
         via scaled dot-product  ->  similarity logits.
      3. Add a LEARNABLE per-ring bias to the logits before softmax, so the
         model can learn to favour inner-ring slots (persistent entities) over
         outer-ring slots (transient). Nothing is hard-coded.
      4. Softmax over slots -> a soft "orbital" probability cloud (NOT top-k).
      5. Retrieve = probability-weighted sum of slots.
      6. Fuse retrieved content back into h with a learned gate (gated add):
             h_out = h + sigmoid(gate(h)) * o_proj(retrieved)

    The whole read is toggled by the caller (use_query_read); when off, h is
    returned untouched, so the control arm is architecturally identical.
    """

    def __init__(self, config):
        super().__init__()
        self.d         = config.model.dim
        self.num_rings = config.model.num_rings

        self.q_proj    = nn.Linear(self.d, self.d, bias=False)
        self.o_proj    = nn.Linear(self.d, self.d, bias=False)
        self.gate_proj = nn.Linear(self.d, self.d, bias=True)

        # learnable per-ring retrieval bias, added to similarity logits.
        # init 0 -> starts as an unbiased read; training decides ring preference.
        self.ring_bias = nn.Parameter(torch.zeros(self.num_rings))

    def forward(self, x, M_all, slot_ring_idx):
        """
        x            : (B, T, d)    hidden states (queries)
        M_all        : (B, S, d)    all memory slots pooled across rings/nodes
        slot_ring_idx: (S,) long    ring index for each slot (for ring bias)

        returns: (x_out (B,T,d), attn (B,T,S))
        """
        q = self.q_proj(x)                                   # (B,T,d)
        M = M_all.to(q.dtype)                                # align dtype (autocast safe)

        scores = torch.einsum("btd,bsd->bts", q, M) / (self.d ** 0.5)   # (B,T,S)
        scores = scores + self.ring_bias[slot_ring_idx].view(1, 1, -1)  # per-ring bias

        # softmax in float32 for stability, cast back to working dtype
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)        # (B,T,S)

        retrieved = torch.einsum("bts,bsd->btd", attn, M)              # (B,T,d)
        gate      = torch.sigmoid(self.gate_proj(x))                   # (B,T,d)
        x_out     = x + gate * self.o_proj(retrieved)
        return x_out, attn


class StructuralOrbitalRead(nn.Module):
    """
    Position-resolved, content-addressable read with an optional STRUCTURAL
    (direction-gated) mask.

    Fixes the time-collapse problem of the slot read: the memory bank here is
    the per-position hidden states themselves (NO mean-pool over time), so a
    query can reach a SPECIFIC earlier token (e.g. the "ball" position).

    Each position is tagged by the ring it settled in. Two masks on the
    similarity logits (set to -inf before softmax):

      - CAUSAL (always): a query at position t may only read positions s <= t,
        so it never sees the future (no answer leakage).
      - STRUCTURAL (only when use_mask=True): a query at ring r may only read
        positions whose ring is INWARD-or-equal, i.e. ring[s] <= ring[t]
        (ring 0 = innermost). Encodes "to recall persistent entities, query
        inward." With use_mask=False this is the FLAT attention-style read.

    A learnable per-ring bias on the keys is kept in BOTH modes, so the two arms
    differ ONLY by the structural mask. retrieved content is gate-fused back in.
    """

    def __init__(self, config):
        super().__init__()
        self.d         = config.model.dim
        self.num_rings = config.model.num_rings
        self.q_proj    = nn.Linear(self.d, self.d, bias=False)
        self.o_proj    = nn.Linear(self.d, self.d, bias=False)
        self.gate_proj = nn.Linear(self.d, self.d, bias=True)
        self.ring_bias = nn.Parameter(torch.zeros(self.num_rings))

    def forward(self, x, ring_of_pos, use_mask):
        """
        x           : (B, T, d)  hidden states = queries AND memory bank
        ring_of_pos : (B, T)     ring index each position settled in
        use_mask    : bool       apply the directional (inward) structural mask
        returns: (x_out (B,T,d), attn (B,T,T), lookback_frac scalar)
        """
        B, T, d = x.shape
        q = self.q_proj(x)
        k = x.to(q.dtype)            # values = keys = position-resolved states
        scores = torch.einsum("btd,bsd->bts", q, k) / (self.d ** 0.5)   # (B,T,T)
        scores = scores.float()
        scores = scores + self.ring_bias[ring_of_pos].unsqueeze(1)       # per-key ring bias

        # causal mask: key s <= query t
        causal  = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        allowed = causal.unsqueeze(0).expand(B, T, T).clone()
        if use_mask:
            rq = ring_of_pos.unsqueeze(2)        # (B,T,1) query ring
            rk = ring_of_pos.unsqueeze(1)        # (B,1,T) key ring
            allowed = allowed & (rk <= rq)       # inward-or-equal only
        scores = scores.masked_fill(~allowed, float("-inf"))

        attn = torch.softmax(scores, dim=-1).to(q.dtype)                 # (B,T,T)
        retrieved = torch.einsum("bts,bsd->btd", attn, k)               # (B,T,d)
        gate  = torch.sigmoid(self.gate_proj(x))
        x_out = x + gate * self.o_proj(retrieved)

        with torch.no_grad():
            self_mass = attn.float().diagonal(dim1=1, dim2=2).mean()     # attn to self
        return x_out, attn, float(1.0 - self_mass)


class HeuristicWriteController:
    """
    Content-similarity ASSIGNMENT controller for the separable-write SRM.

    For each incoming representation v it decides, with NO learning:
      (a) NEW vs REPEAT — compare v to the currently OCCUPIED slots:
            max cosine >= repeat_thresh  -> REPEAT  (route to that slot)
            otherwise                    -> NEW     (allocate the next free slot)
      (b) which slot — the argmax-cosine slot for a repeat, the next free slot
          for a new entity.

    Optional write-gate (use_gate=True): an incoming salience scalar (standing in
    for a learned/lexical entity-detector) below gate_thresh is SKIPPED, so
    non-entity "filler" never consumes a slot. With use_gate=False this is the
    pure content heuristic, which (by design) cannot tell a brand-new entity from
    a brand-new filler — both look novel — so filler will also allocate slots.

    On a repeat the slot is EMA-updated toward v (update in [0,1]) and kept unit-
    norm, so noisy recurrences of the same entity refine — not corrupt — its slot.

    This is a plain (param-free) module; it owns its own (slots, d) bank so the
    test can drive it directly without the full web.
    """

    def __init__(self, dim, slots, repeat_thresh=0.5, gate_thresh=0.5,
                 update=0.5, use_gate=True):
        self.d = dim
        self.S = slots
        self.repeat_thresh = repeat_thresh
        self.gate_thresh = gate_thresh
        self.update = update
        self.use_gate = use_gate
        self.reset()

    def reset(self, device="cpu"):
        self.M = torch.zeros(self.S, self.d, device=device)
        self.occ = torch.zeros(self.S, dtype=torch.bool, device=device)
        self.n_alloc = 0
        return self

    @torch.no_grad()
    def step(self, v, salience=1.0):
        """
        v        : (d,) incoming representation
        salience : scalar entity-detector signal (used only if use_gate)
        returns dict(action, slot, max_cos) with action in
                {"skip","repeat","new","overflow"}.
        """
        if self.use_gate and float(salience) < self.gate_thresh:
            return {"action": "skip", "slot": -1, "max_cos": None}

        if self.occ.any():
            occ_idx = self.occ.nonzero(as_tuple=True)[0]
            sims = F.cosine_similarity(self.M[occ_idx], v.unsqueeze(0), dim=1)
            mx, am = torch.max(sims, dim=0)
            max_cos = mx.item()
            best = occ_idx[am].item()
        else:
            max_cos, best = -1.0, -1

        if max_cos >= self.repeat_thresh:                       # REPEAT -> route + refine
            self.M[best] = F.normalize(
                (1.0 - self.update) * self.M[best] + self.update * v, dim=0)
            return {"action": "repeat", "slot": best, "max_cos": max_cos}

        if self.n_alloc >= self.S:                              # out of slots
            return {"action": "overflow", "slot": -1, "max_cos": max_cos}

        slot = self.n_alloc                                     # NEW -> next free slot
        self.M[slot] = F.normalize(v, dim=0)
        self.occ[slot] = True
        self.n_alloc += 1
        return {"action": "new", "slot": slot, "max_cos": max_cos}


class LearnedWriteController(nn.Module):
    """
    Trainable counterpart to the heuristic controller (this step only confirms it
    RUNS; we are mainly testing the heuristic).

    Given an incoming v and the current bank M it produces:
      - write_gate : sigmoid scalar  ~ P(this is an entity worth writing)
      - slot_logits: similarity of a learned query to each slot (soft routing /
                     repeat detection — argmax is the routed slot).
    """

    def __init__(self, dim, slots):
        super().__init__()
        self.write_gate = nn.Linear(dim, 1)
        self.slot_query = nn.Linear(dim, dim, bias=False)

    def forward(self, v, M):
        """v: (B,d)   M: (slots,d)  ->  gate (B,1), slot_logits (B,slots)"""
        gate = torch.sigmoid(self.write_gate(v))
        q = self.slot_query(v)
        slot_logits = q @ M.t() / (M.shape[-1] ** 0.5)
        return gate, slot_logits


class SeparableMemoryRead(nn.Module):
    """
    Write-side redesign — the three de-risked pieces, integrated end-to-end and
    DIFFERENTIABLY as a causal, content-routed, gated entity memory. This is the
    post-hop ENTITY store (the hop-loop SRM stays as local working memory).

    Realised pieces
    ---------------
      * learned write-gate (small MLP)  -> g_u in (0,1): "is position u an entity
        worth storing?" (entity-vs-filler). MLP, not linear, so it can model a
        nonlinear entity-ness manifold (the caveat from the gate de-risk test).
      * content-similarity assignment    -> every position routes to LEARNED slot
        addresses by softmax(content . slot_addr). Same-content positions (an
        entity and its later mentions) route to the SAME slot; distinct entities
        to different slots — the new-vs-repeat controller, made differentiable.
      * separable (non-blending) write   -> a slot accumulates only the positions
        routed to it; positions routed elsewhere never blend in. Reading a slot
        returns the gate-weighted average of what was written there.

    Causality: position t may bind only to entities written at u <= t.

    Efficiency (no (B,T,S,d) tensor): with write_w, read_w the soft slot
    assignments (softmax over S slots) and g the writer gate, the cumulative
    separable memory read collapses to a causal attention whose scores are the
    low-rank slot-routing agreement:

        retrieved_t = sum_{u<=t} C[t,u] * g_u * value_u  /  (sum_{u<=t} C[t,u]*g_u)
        with  C[t,u] = < read_w[t] , write_w[u] >  in [0,1]

    i.e. position t reads strongly from earlier positions routed to the SAME
    slot(s). Memory is implicit; only (B,T,T) and (B,T,S) tensors are formed.

    Warm-start: o_proj is zero-initialised and the fusion gate biased closed, so
    at init the module is the identity (x_out == x) and a warm restart keeps the
    baseline CE. Training opens the gate.
    """

    def __init__(self, config):
        super().__init__()
        d = config.model.dim
        S = config.memory.slots
        self.d, self.S = d, S

        self.write_key  = nn.Linear(d, d, bias=False)   # write-routing key
        self.read_query = nn.Linear(d, d, bias=False)   # read-routing query
        self.value      = nn.Linear(d, d, bias=False)   # stored payload
        self.slot_addr  = nn.Parameter(torch.randn(S, d) / (d ** 0.5))

        hg = max(16, d // 2)
        self.gate_mlp = nn.Sequential(                  # learned write-gate (MLP)
            nn.Linear(d, hg), nn.GELU(), nn.Linear(hg, 1),
        )

        self.o_proj    = nn.Linear(d, d, bias=False)    # fuse retrieved -> residual
        self.out_gate  = nn.Linear(d, d, bias=True)
        self.route_temp = 0.5                           # <1 sharpens slot routing

        # ── Step 6 write policies (config-gated; no-op defaults) ────────────────
        self.write_decay = float(getattr(config.model, "write_decay", 1.0))  # γ (1=off)
        self.erase       = bool(getattr(config.model, "erase", False))
        if self.erase:
            # learned erase gate e = sigmoid(erase_gate(x)); zero weight + very
            # negative bias -> e ~ 0 at warm-start (erase OFF, warm-compatible).
            self.erase_gate = nn.Linear(d, 1)
            nn.init.zeros_(self.erase_gate.weight)
            nn.init.constant_(self.erase_gate.bias, -6.0)

        # identity init -> warm-start CE == baseline; training opens it up
        nn.init.zeros_(self.o_proj.weight)
        nn.init.constant_(self.out_gate.bias, -4.0)

    def forward(self, x, causal=True, subj_id=None, oracle_bind=False,
                id_key=None, oracle_alpha=1.0, giver_key_in=None):
        """x: (B,T,d) -> (x_out (B,T,d), stats dict).

        id_key (identity-key addressing): when provided (B,T,d), the slot
        write_key/read_query are computed from THIS tensor (the raw token
        EMBEDDING, identity-preserving) instead of the post-mixing hidden state
        x where subject identity is smeared. The VALUE path still reads x, so
        only the ADDRESS is re-sourced. This gives write and read a stable
        discrete anchor to agree on, so the learned router can form a
        subject-keyed address on its own (no oracle at inference).

        oracle_bind (localization probe / anneal teacher): where subj_id[b,t]>=0,
        blend the learned content-routed write_w/read_w toward a one-hot on
        (subj_id % S) with weight oracle_alpha in [0,1]:
            w <- alpha * one_hot + (1-alpha) * w_learned
        alpha=1.0 (default) == the hard oracle override (backward-compatible);
        annealing alpha 1->0 hands routing back to the learned router while the
        value/emit path stays trained (distillation from the oracle teacher).
        stats exposes write_w/read_w (on-graph) so training can supervise them
        toward subj_id % S (L_addr) with the oracle OFF."""
        B, T, d = x.shape
        scale = (d ** 0.5) * self.route_temp

        key_in = id_key if id_key is not None else x    # identity-preserving addressing source
        wk = self.write_key(key_in)
        rq = self.read_query(key_in)
        write_w = torch.softmax((wk @ self.slot_addr.t()) / scale, dim=-1)   # (B,T,S)
        read_w  = torch.softmax((rq @ self.slot_addr.t()) / scale, dim=-1)   # (B,T,S)

        if oracle_bind and subj_id is not None:
            sel  = (subj_id >= 0).unsqueeze(-1)                              # (B,T,1)
            slot = (subj_id.clamp_min(0) % self.S)                          # (B,T)
            oh   = F.one_hot(slot, self.S).to(write_w.dtype)               # (B,T,S)
            a    = float(oracle_alpha)                                      # 1=hard oracle, 0=learned
            write_w = torch.where(sel, a * oh + (1.0 - a) * write_w, write_w)  # subject-keyed write
            read_w  = torch.where(sel, a * oh + (1.0 - a) * read_w,  read_w)   # subject-keyed read

        gate   = torch.sigmoid(self.gate_mlp(x)).squeeze(-1)                 # (B,T)
        values = self.value(x)                                              # (B,T,d)

        # per-position write STRENGTH to each slot: a[u,s] = write_w[u,s] * gate[u]
        a = write_w * gate.unsqueeze(-1)                                    # (B,T,S)

        gamma = self.write_decay
        erase_on = self.erase and giver_key_in is not None
        if gamma >= 1.0 and not erase_on:
            # ── Step-5 (additive) path: exact, no decay / no erase ──────────────
            # slot-routing agreement C[t,u] = read_w[t] . write_w[u]  in [0,1]
            C = torch.einsum("bts,bus->btu", read_w, write_w)               # (B,T,T)
            W = C * gate.unsqueeze(1)                                       # weight by writer gate
            erase_stats = None
        else:
            # ── Step-6 (decayed / erased) path ──────────────────────────────────
            # An explicit decayed slot memory M_s(t) = Σ_{u<=t} a[u,s] value[u] ·
            # Π_{u<v<=t} D[v,s] read as retrieved[t]=Σ_s read_w[t,s] M_s(t) has the
            # EXACT closed form  W'[t,u] = Σ_s read_w[t,s]·a[u,s]·exp(L[t,s]-L[u,s])
            # with  L[t,s] = Σ_{v<=t} log D[v,s]  the cumulative log-decay. γ=1 &
            # erase off -> D=1, L=0 -> W'==C·gate (identical to the path above), so
            # this is a strict generalization that keeps the per-position read_dist.
            logD = torch.zeros(B, T, self.S, device=x.device, dtype=torch.float32)
            if gamma < 1.0:                                                 # DECAY-ON-WRITE
                logD = logD + torch.log((1.0 - (1.0 - gamma) * a).clamp_min(1e-6))
            erase_stats = None
            if erase_on:                                                   # TARGETED ERASE
                # giver slot address from the SECOND lookback's name key; e is the
                # learned erase gate (zero-init -> ~0). slot <- slot*(1 - e*w_giver)
                # is one extra decay event per position, folded into logD.
                gk = self.write_key(giver_key_in)                          # reuse write-routing
                w_giver = torch.softmax((gk @ self.slot_addr.t()) / scale, dim=-1)  # (B,T,S)
                e = torch.sigmoid(self.erase_gate(x)).squeeze(-1)          # (B,T) in (0,1)
                logD = logD + torch.log((1.0 - e.unsqueeze(-1) * w_giver).clamp_min(1e-6))
                erase_stats = {"e": e, "w_giver": w_giver}
            L = torch.cumsum(logD, dim=1).clamp_min(-60.0)                 # (B,T,S), <=0
            RT = read_w * torch.exp(L)                                     # read side  @ t
            AU = a * torch.exp(-L)                                         # write side @ u
            W = torch.einsum("bts,bus->btu", RT, AU)                       # (B,T,T) = W'
            # gate is already folded into a (hence W); no separate ·gate needed.

        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            W = W.masked_fill(~mask.unsqueeze(0), 0.0)

        denom     = W.sum(dim=-1, keepdim=True) + 1e-6
        # read_dist[t,u] = P(position t reads source position u) via shared-slot
        # routing, gated by writer entity-ness, causal. This is the orbital
        # memory's OWN content-addressed read over source positions — kept on the
        # autograd graph so it can both (a) feed a memory-sourced copy and (b) be
        # supervised toward recall->source (the RETRIEVE link). NOT an attn map.
        # Under Step-6 policies W already carries per-write decay + giver-slot erase.
        read_dist = W / denom                                               # (B,T,T)
        retrieved = torch.einsum("btu,bud->btd", read_dist, values)         # (B,T,d)

        fuse  = torch.sigmoid(self.out_gate(x))
        x_out = x + fuse * self.o_proj(retrieved)

        with torch.no_grad():
            ent = -(write_w.clamp_min(1e-9).log() * write_w).sum(-1).mean()  # routing entropy
            stats = {
                "gate_mean":     float(gate.mean()),
                "gate_frac_on":  float((gate > 0.5).float().mean()),
                "route_entropy": float(ent),
                "fuse_mean":     float(fuse.mean()),
            }
        stats["read_dist"] = read_dist          # (B,T,T) on-graph: memory read
        stats["write_w"]   = write_w            # (B,T,S) on-graph: slot routing (write) — L_addr target
        stats["read_w"]    = read_w             # (B,T,S) on-graph: slot routing (read)  — L_addr target
        if erase_stats is not None:             # Step 6: erase gate e + giver address (on-graph)
            stats["erase_e"]  = erase_stats["e"]        # (B,T)
            stats["w_giver"]  = erase_stats["w_giver"]  # (B,T,S)
        return x_out, stats