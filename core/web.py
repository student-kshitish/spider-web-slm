import torch
import torch.nn as nn
import torch.nn.functional as F
from core.node import WebNode
from core.rope import SpiderWebRoPE
from core.memory import OrbitalQueryRead, StructuralOrbitalRead, SeparableMemoryRead
from core.hybrid import HybridLookbackAttention, NameLookback


class SpiderWeb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.d = config.model.dim

        self.embed = nn.Embedding(config.model.vocab_size, self.d)
        self.rope  = SpiderWebRoPE(config)

        self.rings = nn.ModuleList([
            nn.ModuleList([
                WebNode(config) for _ in range(config.model.nodes_per_ring)
            ])
            for _ in range(config.model.num_rings)
        ])

        # Substrate fix (3): drop spectral_norm on the head when sharp_head is set,
        # so logits can spike rare (entity) tokens above the frequency prior.
        # NOTE: lm_head is RETAINED as the OFF-path / control readout. When
        # use_pointer is set, the generative branch instead ties to embed.weight
        # (every token, incl. unseen entities, gets an output direction) and lm_head
        # is left unused — so use_pointer=OFF reproduces current behaviour exactly.
        _head = nn.Linear(self.d, config.model.vocab_size, bias=False)
        self.lm_head = (_head if getattr(config.model, "sharp_head", False)
                        else nn.utils.spectral_norm(_head))
        self.final_norm = nn.RMSNorm(self.d)

        # ── Pointer / copy readout gate (only used when config.model.use_pointer) ──
        # lambda = sigmoid(copy_gate(x)) is the weight on the GENERATIVE branch;
        # (1-lambda) weights the COPY branch. Zero-init bias -> neutral lambda~0.5
        # at warm-start (gate starts active, free to move either way, not collapsed).
        self.copy_gate = nn.Linear(self.d, 1)
        nn.init.zeros_(self.copy_gate.bias)
        # ── conc_gate (bounded experiment): re-source the gate from READ-CONCENTRATION
        # stats [read_max, read_entropy] (lexicon-invariant) instead of / plus x. Only
        # built when config.model.conc_gate in {"stats","stats_x"}; zero-bias so lambda
        # starts neutral. copy_gate above stays as the "off" path. See config.py.
        self._conc_gate = getattr(config.model, "conc_gate", "off")
        if self._conc_gate in ("stats", "stats_x"):
            in_dim = 2 + (self.d if self._conc_gate == "stats_x" else 0)
            self.conc_gate = nn.Linear(in_dim, 1)
            nn.init.zeros_(self.conc_gate.weight)   # neutral lambda~0.5 at warm-start
            nn.init.zeros_(self.conc_gate.bias)

        # ── Radial hierarchy: inner-ring entity recall ───────────────────────
        # Projects the mean inner-ring memory slot back to embedding space so
        # the model can be trained to reconstruct a lagged token embedding.
        self.recall_proj = nn.Linear(self.d, self.d, bias=False)
        self._recall_lag = 4   # reconstruct token seen this many positions ago

        # ── Orbital query read: content-addressable retrieval over all slots ──
        self.query_read = OrbitalQueryRead(config)

        # ── Structural read: position-resolved, direction-gated retrieval ─────
        self.struct_read = StructuralOrbitalRead(config)

        # ── Separable entity memory (write-side redesign) ─────────────────────
        # Causal, content-routed, gated non-blending entity store. Active only
        # when config.memory.write_mode == "separable"; "blend" == baseline.
        self.separable_mem = SeparableMemoryRead(config)

        # ── Unified hybrid: surgical gated lookback attention ─────────────────
        # Spider Web rings (above) keep routing/computing for ALL tokens. This
        # module adds content-based causal attention ONLY for gate-flagged
        # tokens, fused back gate-weighted. Active only when use_hybrid is set;
        # warm-start safe (o_proj zero-init -> identity at init).
        self.hybrid_lookback = HybridLookbackAttention(
            config, lookback_width=getattr(config.model, "lookback_width", 32)
        )

        # ── Name-lookback (Step 2): learned governing-name locator ─────────────
        # Replaces the name_transport oracle. Locates the governing name token and
        # transports its VERBATIM embedding to serve as the separable-memory slot
        # key (id_key). Active only when config.model.name_lookback; its attention
        # is supervised toward name_src in training, but name_src NEVER enters the
        # forward — at inference it must locate the name unaided.
        self.name_lookback = NameLookback(config)

    def forward(self, input_ids, tau=1.0, hard=False, return_ring_stats=False,
                use_query_read=None, use_struct_read=None, struct_mask=None,
                use_hybrid=None, lookback_width="default", use_pointer=None,
                subj_id=None, oracle_alpha=1.0, name_src=None):
        B, T   = input_ids.shape
        device = input_ids.device
        NR     = self.cfg.model.num_rings
        NN     = self.cfg.model.nodes_per_ring

        # ── EMBEDDING + ROPE ─────────────────────────────────────────────────
        x = self.embed(input_ids)
        x = self.rope(x, node_idx=0, ring_idx=NR - 1)

        # Substrate fix (4): when no_meanpool is set, the per-node memory is kept
        # POSITION-RESOLVED (B,T,slots,d) instead of being mean-pooled across the
        # T axis at write time (web.py:172). An entity written at position k then
        # survives in that position's own bank instead of being averaged into an
        # irreversible global summary.
        no_meanpool = getattr(self.cfg.model, "no_meanpool", False)
        S = self.cfg.memory.slots

        # ── MEMORY REGISTRY  (B, slots, d)  |  (B, T, slots, d) if no_meanpool ──
        m_t_registry = [
            [self.rings[r][n].memory.init_memory(B, device)[0]
             for n in range(NN)]
            for r in range(NR)
        ]
        if no_meanpool:
            m_t_registry = [
                [m_t_registry[r][n].unsqueeze(1).expand(B, T, S, self.d).contiguous()
                 for n in range(NN)]
                for r in range(NR)
            ]

        current_ring = torch.full((B, T), NR - 1, dtype=torch.long, device=device)
        active_mask  = torch.ones((B, T), dtype=torch.bool, device=device)

        all_aux_logits = []
        all_routings   = []
        spatial_chain  = [None] * NR

        # Radial hierarchy losses (scalars accumulated over the hop loop)
        depth_loss  = torch.tensor(0.0, device=device)   # (a) outer-ring exit pressure
        recall_loss = torch.tensor(0.0, device=device)   # (b) inner-ring entity recall
        recall_cnt  = 0

        # Ring stats accumulators (populated only when return_ring_stats=True)
        _rs_visits    = {r: 0   for r in range(NR)}
        _rs_routing   = {r: torch.zeros(3, device=device) for r in range(NR)}
        _rs_r_count   = {r: 0   for r in range(NR)}
        _rs_h_norm    = {r: 0.0 for r in range(NR)}
        _rs_h_count   = {r: 0   for r in range(NR)}
        _rs_exit_ring = {r: 0   for r in range(NR)}

        # ── HOP LOOP ─────────────────────────────────────────────────────────
        for hop in range(self.cfg.routing.max_hops):
            new_x = x.clone()

            for r_idx in reversed(range(NR)):
                ring      = self.rings[r_idx]
                ring_mask = (current_ring == r_idx) & active_mask

                if not ring_mask.any():
                    continue

                for n_idx, node in enumerate(ring):

                    t_idx = torch.arange(T, device=device)
                    node_mask = ring_mask & (t_idx % NN == n_idx)

                    if not node_mask.any():
                        continue

                    # ── TOKEN GATHER ─────────────────────────────────────────
                    flat_mask = node_mask.reshape(-1)
                    x_flat    = x.reshape(B * T, self.d)

                    h_in       = x_flat[flat_mask]
                    num_active = h_in.size(0)

                    # ── NEIGHBOURS ───────────────────────────────────────────
                    ring_flat   = ring_mask.reshape(-1)
                    ring_states = x_flat[ring_flat]

                    if ring_states.size(0) > 0:
                        mean       = ring_states.mean(0, keepdim=True)
                        neighbours = mean.unsqueeze(1).expand(num_active, 1, self.d)
                    else:
                        neighbours = torch.zeros(num_active, 1, self.d, device=device)

                    # ── MEMORY (TOKEN-ALIGNED) ────────────────────────────────
                    m_t = m_t_registry[r_idx][n_idx]
                    if no_meanpool:
                        # (B,T,slots,d) — each position reads its OWN bank, no expand
                        m_t_flat = m_t.reshape(B * T, S, self.d)
                    else:
                        # (B,slots,d) — global bank shared across all positions
                        m_t_exp  = m_t.unsqueeze(1).expand(B, T, -1, -1)
                        m_t_flat = m_t_exp.reshape(B * T, m_t.size(1), self.d)
                    m_t_in   = m_t_flat[flat_mask]            # (N, slots, d)

                    # ── SPATIAL MEMORY ────────────────────────────────────────
                    m_s = None
                    if r_idx + 1 < NR and spatial_chain[r_idx + 1] is not None:
                        m_s_full = spatial_chain[r_idx + 1]
                        m_s_exp  = m_s_full.unsqueeze(1).expand(B, T, -1, -1)
                        m_s_flat = m_s_exp.reshape(B * T, m_s_full.size(1), self.d)
                        m_s      = m_s_flat[flat_mask]

                    # ── NODE FORWARD ──────────────────────────────────────────
                    out = node(h_in, neighbours, m_t_in, m_s, tau=tau, hard=hard)

                    # ── WRITE BACK HIDDEN ────────────────────────────────────
                    # Substrate fix (2): additive residual highway. Instead of
                    # REPLACING the position's state with the node output, ADD it,
                    # so the original token representation is carried forward across
                    # hops (x = x + node_out) and survives to the recall position.
                    new_x_flat = new_x.reshape(B * T, self.d)
                    if getattr(self.cfg.model, "residual_stream", False):
                        new_x_flat[flat_mask] = x_flat[flat_mask] + out["h"]
                    else:
                        new_x_flat[flat_mask] = out["h"]
                    new_x = new_x_flat.reshape(B, T, self.d)

                    # ── WRITE BACK MEMORY ─────────────────────────────────────
                    # Substrate fix (1): when undetach_mem is set, keep m_t on the
                    # graph so gradient flows from a downstream READ back to this
                    # WRITE (the model can learn WHAT to store). Default detaches
                    # (original non-differentiable scratchpad behaviour).
                    m_t_flat_full = m_t_flat.clone()
                    m_write = (out["m_t"] if getattr(self.cfg.model, "undetach_mem", False)
                               else out["m_t"].detach())
                    m_t_flat_full[flat_mask] = m_write.to(m_t_flat_full.dtype)
                    if no_meanpool:
                        # Substrate fix (4): keep the T axis — position-resolved store.
                        # The entity written at this position is NOT averaged away.
                        m_t_new = m_t_flat_full.reshape(B, T, S, self.d)
                    else:
                        # baseline: collapse T -> (B,slots,d) global summary (the
                        # irreversible mean the probe localised as the destroyer).
                        m_t_new = m_t_flat_full.reshape(B, T, S, self.d).mean(dim=1)
                    m_t_registry[r_idx][n_idx] = m_t_new

                    # ── ROUTING ───────────────────────────────────────────────
                    routing   = out["routing"]          # (N, 3): [stay, move_in, exit]
                    decisions = routing.argmax(dim=-1)

                    idxs    = flat_mask.nonzero(as_tuple=True)[0]
                    cr_flat = current_ring.reshape(-1)
                    am_flat = active_mask.reshape(-1)

                    # move inward (only if not already at innermost ring)
                    move_in = (decisions == 1) & (cr_flat[idxs] > 0)
                    cr_flat[idxs[move_in]] -= 1

                    # FIXED exit: only explicit exit decision (action 2).
                    # Previously `| (cr == 0)` caused ring-0 to be unreachable.
                    exit_mask = decisions == 2
                    am_flat[idxs[exit_mask]] = False

                    current_ring = cr_flat.reshape(B, T)
                    active_mask  = am_flat.reshape(B, T)

                    all_aux_logits.append(out["aux_logits"])
                    all_routings.append(routing)

                    # ── (a) DEPTH PRESSURE LOSS (outer rings) ────────────────
                    # Penalise choosing exit over move_in for outer rings.
                    # Pushes tokens inward rather than bailing out early.
                    # Weight w_depth is applied in loss.py (default 0.0).
                    if r_idx >= NR // 2:
                        depth_loss = depth_loss + (
                            routing[:, 2] - routing[:, 1]
                        ).mean()

                    # ── (b) INNER-RING RECALL LOSS ───────────────────────────
                    # Inner ring memory must reconstruct the embedding of a
                    # token seen `lag` positions back.  This trains the SRM to
                    # hold entity state rather than transient token detail.
                    # Weight w_recall applied in loss.py (default 0.0).
                    if r_idx < NR // 2 and T > self._recall_lag:
                        bt_indices = flat_mask.nonzero(as_tuple=True)[0]
                        b_idx      = bt_indices // T
                        t_pos      = bt_indices % T
                        valid      = t_pos >= self._recall_lag
                        if valid.any():
                            target_embs = self.embed(
                                input_ids[b_idx[valid], t_pos[valid] - self._recall_lag]
                            ).detach()
                            m_summary   = out["m_t"][valid].mean(1)   # (M, d)
                            pred        = self.recall_proj(m_summary)
                            # cosine reconstruction loss in [-1, 1] space (stable)
                            pred_n   = F.normalize(pred.float(),         dim=-1)
                            target_n = F.normalize(target_embs.float(),  dim=-1)
                            recall_loss = recall_loss + (
                                1.0 - (pred_n * target_n).sum(-1)
                            ).mean()
                            recall_cnt += 1

                    # ── RING STATS ────────────────────────────────────────────
                    if return_ring_stats:
                        with torch.no_grad():
                            _rs_visits[r_idx]  += flat_mask.sum().item()
                            _rs_routing[r_idx] += routing.float().mean(0)
                            _rs_r_count[r_idx] += 1
                            _rs_h_norm[r_idx]  += h_in.float().norm(dim=-1).mean().item()
                            _rs_h_count[r_idx] += 1
                            exited_idxs = idxs[exit_mask]
                            for idx_val in exited_idxs:
                                ring_val = cr_flat[idx_val].item()
                                if ring_val in _rs_exit_ring:
                                    _rs_exit_ring[ring_val] += 1

                # ── SPATIAL CHAIN UPDATE ─────────────────────────────────────
                ring_flat_all = ring_mask.reshape(-1)
                if ring_flat_all.any():
                    ring_h = new_x.reshape(B * T, self.d)[ring_flat_all]
                    slots  = self.cfg.memory.slots
                    mean   = ring_h.mean(0)
                    spatial_chain[r_idx] = mean.unsqueeze(0).unsqueeze(0).expand(
                        B, slots, self.d
                    ).contiguous()

            x = new_x

            if not active_mask.any():
                break

        # ── ORBITAL QUERY READ (content-addressable retrieval) ────────────────
        # Pool the final per-node memory slots across all rings/nodes, then let
        # each position query them. Toggleable: when off, x is untouched so the
        # control arm is architecturally identical.
        use_qr = (self.cfg.model.use_query_read if use_query_read is None
                  else use_query_read)
        ring_read_mass = None
        if use_qr:
            if no_meanpool:
                raise NotImplementedError(
                    "use_query_read + no_meanpool: the memory registry is "
                    "position-resolved (B,T,slots,d); the slot-pooling this read "
                    "assumes (B,slots,d) is undefined. Not used in this run.")
            slot_list, ring_ids = [], []
            for r in range(NR):
                for n in range(NN):
                    mem = m_t_registry[r][n]          # (B, slots, d)
                    slot_list.append(mem)
                    ring_ids.extend([r] * mem.size(1))
            M_all         = torch.cat(slot_list, dim=1)                       # (B, S, d)
            slot_ring_idx = torch.tensor(ring_ids, device=device, dtype=torch.long)
            x, qr_attn = self.query_read(x, M_all, slot_ring_idx)

            # realized per-ring retrieval mass (how much weight each ring got)
            with torch.no_grad():
                avg = qr_attn.float().mean(dim=(0, 1))                        # (S,)
                ring_read_mass = torch.zeros(NR, device=device)
                ring_read_mass.index_add_(0, slot_ring_idx, avg)

        # ── STRUCTURAL READ (position-resolved, direction-gated) ──────────────
        # Memory bank = per-position hidden states (NO time-collapse). Each
        # position tagged by its final ring (current_ring). Causal always;
        # directional inward mask when struct_mask is on. Toggleable.
        use_sr = (self.cfg.model.use_struct_read if use_struct_read is None
                  else use_struct_read)
        struct_lookback = None
        if use_sr:
            sm = (self.cfg.model.struct_mask if struct_mask is None else struct_mask)
            x, _sr_attn, struct_lookback = self.struct_read(x, current_ring, sm)

        # ── SEPARABLE ENTITY MEMORY (write-side redesign) ─────────────────────
        # Causal, content-routed, gated non-blending store. The A/B flag:
        #   write_mode == "separable" -> active     write_mode == "blend" -> off (baseline)
        write_mode = getattr(self.cfg.memory, "write_mode", "blend")
        sep_stats = None
        name_stats = None                 # name-lookback attn/name_hat, or None (off)
        if write_mode == "separable":
            ob = getattr(self.cfg.model, "oracle_bind", False)
            # IDENTITY-KEY ADDRESSING: feed the slot key/query from the raw token
            # EMBEDDING (identity-preserving: pre-RoPE, pre-ring-mixing) instead of
            # the smeared hidden state x. self.embed(input_ids) is the SAME table
            # used at forward entry (line 80) before RoPE — a stable per-token
            # anchor for write and read to agree on. Value path still uses x.
            id_key = (self.embed(input_ids)
                      if getattr(self.cfg.model, "id_key_addr", False) else None)
            # NAME-TRANSPORT ORACLE: replace the key at labeled write/read positions
            # with the embedding of the governing NAME token (name_src[b,t] = that
            # token's position; -1 = keep default). This corrects the key-source bug:
            # the object-intro/recall positions hold "ball"/"the", which carry no
            # subject identity — here we gather the SUBJECT-/CUE-name embedding instead.
            if (id_key is not None and name_src is not None
                    and getattr(self.cfg.model, "name_transport", False)):
                sel  = (name_src >= 0).unsqueeze(-1)                       # (B,T,1)
                gidx = name_src.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.d)
                gathered = torch.gather(id_key, 1, gidx)                   # embed at name pos
                id_key = torch.where(sel, gathered, id_key)
            # NAME-LOOKBACK (Step 2): LEARN to locate the governing name instead of
            # gathering it by oracle position. name_hat[t] = sum_u a[t,u]*embed(u) is
            # an identity-preserving transport of raw embeddings; it BECOMES the slot
            # key (id_key). name_src is NOT read here — the lookback attention is
            # supervised toward it externally (train only), so at inference the module
            # locates the name on its own. Overrides the id_key source entirely.
            if getattr(self.cfg.model, "name_lookback", False):
                tok_emb = self.embed(input_ids)                            # raw embeddings (value)
                name_hat, name_stats = self.name_lookback(x, tok_emb)
                id_key = name_hat                                          # learned lookback OUTPUT feeds the key
                name_stats["name_hat"] = name_hat                          # on-graph; == id_key (wiring check)
            x, sep_stats = self.separable_mem(x, causal=True,
                                              subj_id=subj_id, oracle_bind=ob,
                                              id_key=id_key, oracle_alpha=oracle_alpha)

        # ── UNIFIED HYBRID: surgical gated lookback attention ─────────────────
        # Spider Web routing above is untouched; this adds gated content
        # attention for flagged tokens only. use_hybrid=False -> pure baseline.
        use_hyb = (getattr(self.cfg.model, "use_hybrid", False)
                   if use_hybrid is None else use_hybrid)
        hybrid_stats = None
        if use_hyb:
            x, hybrid_stats = self.hybrid_lookback(x, width=lookback_width)

        # ── FINAL ─────────────────────────────────────────────────────────────
        x      = self.final_norm(x)
        logits = self.lm_head(x)

        # ── POINTER / COPY READOUT (flag-gated; OFF == lm_head + CE above) ─────
        # Predicted fix for the emit-generalization gap: the per-token lm_head has
        # no output direction for entities it never emitted in training, so even a
        # perfectly-pointed lookback can't decode an unseen noun. The copy branch
        # sidesteps the readout entirely — it emits the token ACTUALLY PRESENT at
        # the position the attention pointed at, using zero per-token emit params:
        #     P_copy[b,t,v] = sum_u attn[b,t,u] * 1[input_ids[b,u] == v]
        # The generative branch is tied to the embedding table so every token has
        # an output direction. A learned gate mixes them; loss is NLL on the
        # mixture (loss.py), since P_copy is a probability, not a logit.
        use_ptr = (getattr(self.cfg.model, "use_pointer", False)
                   if use_pointer is None else use_pointer)
        pointer = None
        if use_ptr:
            V = self.cfg.model.vocab_size
            mem_copy = getattr(self.cfg.model, "mem_copy", False)
            if not mem_copy and (hybrid_stats is None or "attn" not in hybrid_stats):
                raise ValueError(
                    "use_pointer requires use_hybrid (reads hybrid_stats['attn'] "
                    "as the copy distribution). Enable use_hybrid.")
            if mem_copy and (sep_stats is None or "read_dist" not in sep_stats):
                raise ValueError(
                    "mem_copy requires write_mode='separable' (the copy is sourced "
                    "from the orbital memory read sep_stats['read_dist']).")

            # generative branch. FIX A (pointer_warm_gen): read the TRAINED lm_head
            # (already computed as `logits` above) so generation is competent from
            # step 0 (CE ~2.7, not the ~17 of the untrained embed.weight tie) and the
            # gate has no incentive to abandon it. Flag OFF -> original embed tie.
            warm_gen = getattr(self.cfg.model, "pointer_warm_gen", False)
            gen_logits = logits if warm_gen else F.linear(x, self.embed.weight)  # (B,T,V)
            P_gen = F.softmax(gen_logits.float(), dim=-1)          # (B,T,V), sums to 1

            if mem_copy:
                # ── COPY FROM SPIDER WEB'S OWN ORBITAL MEMORY (no attn map) ──────
                # The content-keyed slot-routing read of the separable entity store
                # (read_dist) selects WHICH earlier source position the recall slot
                # points at. We transport that source position's VERBATIM token
                # EMBEDDING through it (identity-preserving), reconstructing the
                # bound entity's vector in embedding space:
                #     retrieved[t] = sum_u read_dist[t,u] * embed(input_ids[u])
                # EMIT: tied-embedding similarity readout (nearest-neighbour over the
                # embedding table, cosine * scale), so unseen entities still decode.
                copy_attn = sep_stats["read_dist"].float()                 # (B,T,u)
                tok_emb   = self.embed(input_ids).float()                  # (B,T,d) verbatim
                retrieved = torch.einsum("btu,bud->btd", copy_attn, tok_emb)  # (B,T,d)
                rn  = F.normalize(retrieved, dim=-1)
                en  = F.normalize(self.embed.weight.float(), dim=-1)       # (V,d)
                scale  = float(getattr(self.cfg.model, "mem_copy_scale", 12.0))
                P_copy = F.softmax(scale * F.linear(rn, en), dim=-1)       # (B,T,V)
            else:
                # copy branch: scatter the (on-graph) attention mass into vocab bins
                # by source-token id. attn rows sum to 1 over u, so P_copy sums to 1.
                copy_attn = hybrid_stats["attn"].float()                  # (B,T,u)
                P_copy = torch.zeros(B, T, V, device=device, dtype=copy_attn.dtype)
                src_ids = input_ids.unsqueeze(1).expand(B, T, copy_attn.size(-1))
                P_copy.scatter_add_(2, src_ids, copy_attn)                # (B,T,V), sums to 1

            # learned copy gate: lambda weights generative, (1-lambda) weights copy.
            # conc_gate: re-source lambda from the copy distribution's CONCENTRATION
            # (read_max, read_entropy) — lexicon-invariant, unlike copy_gate(x). Stats
            # are taken from copy_attn (the on-graph distribution feeding the copy), so
            # gradient from L_router reaches conc_gate's weights (and the read).
            gate_feats = None
            if self._conc_gate in ("stats", "stats_x"):
                ca = copy_attn.float().clamp_min(1e-9)               # (B,T,u)
                read_max = ca.max(dim=-1).values                    # (B,T)
                read_ent = -(ca * ca.log()).sum(dim=-1)             # (B,T)
                gate_feats = torch.stack([read_max, read_ent], dim=-1)  # (B,T,2)
                if self._conc_gate == "stats_x":
                    gate_feats = torch.cat([gate_feats, x.float()], dim=-1)  # (B,T,2+d)
                lam = torch.sigmoid(self.conc_gate(gate_feats).float()).squeeze(-1)
            else:
                lam = torch.sigmoid(self.copy_gate(x).float()).squeeze(-1)    # (B,T)
            # FIX B (pointer_lambda_floor): affine-remap lam into [floor, 1] so the
            # generative branch always keeps gradient and the gate can't fully
            # collapse onto copy. Keeps gradient everywhere (unlike a hard clamp).
            floor = float(getattr(self.cfg.model, "pointer_lambda_floor", 0.0))
            if floor > 0.0:
                lam = floor + (1.0 - floor) * lam
            P = lam.unsqueeze(-1) * P_gen + (1.0 - lam).unsqueeze(-1) * P_copy

            pointer = {
                "P":          P,        # (B,T,V) mixture probability, sums to 1
                "P_gen":      P_gen,    # (B,T,V)
                "P_copy":     P_copy,   # (B,T,V)
                "copy_attn":  copy_attn,  # (B,T,u) distribution feeding the copy
                "copy_src":   "memory" if mem_copy else "attn",
                "lambda":     lam,      # (B,T) gate, on-graph
                "lambda_mean":     float(lam.detach().mean()),
                "lambda_frac_gen": float((lam.detach() > 0.5).float().mean()),
                "copy_mass_mean":  float((1.0 - lam.detach()).mean()),
                "gate_src":        self._conc_gate,   # "off" | "stats" | "stats_x"
                "gate_feats":      gate_feats,        # (B,T,2[+d]) conc-gate input, or None
            }

        # normalise accumulated losses by the number of contributing ring/hops
        depth_loss  = depth_loss  / max(1, NR - NR // 2)   # outer ring count
        recall_loss = recall_loss / max(1, recall_cnt)

        result = {
            "logits":      logits,
            "aux_logits":  all_aux_logits,
            "routings":    all_routings,
            "depth_loss":  depth_loss,
            "recall_loss": recall_loss,
            "use_query_read": use_qr,
            "ring_read_bias": self.query_read.ring_bias.detach(),
            "ring_read_mass": ring_read_mass,   # (NR,) realized retrieval mass, or None
            "use_struct_read":  use_sr,
            "struct_read_bias": self.struct_read.ring_bias.detach(),
            "struct_lookback":  struct_lookback,  # frac of attn mass off-self, or None
            "write_mode":       write_mode,
            "sep_stats":        sep_stats,        # gate/routing stats, or None (blend)
            "name_stats":       name_stats,       # name-lookback attn + name_hat, or None
            "use_hybrid":       use_hyb,
            "hybrid_stats":     hybrid_stats,     # lookback gate stats, or None (off)
            "use_pointer":      use_ptr,
            "pointer":          pointer,          # copy/gen mixture dict, or None (off)
        }

        if return_ring_stats:
            ring_stats    = {}
            total_visits  = sum(_rs_visits.values()) or 1
            for r in range(NR):
                rc = max(_rs_r_count[r], 1)
                hc = max(_rs_h_count[r], 1)
                ring_stats[r] = {
                    "visits":       _rs_visits[r],
                    "visit_frac":   _rs_visits[r] / total_visits,
                    "routing_mean": (_rs_routing[r] / rc).tolist(),
                    "h_norm_mean":  _rs_h_norm[r] / hc,
                    "exit_count":   _rs_exit_ring[r],
                }
            result["ring_stats"] = ring_stats

        return result
