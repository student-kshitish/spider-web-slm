import torch
import torch.nn as nn
from core.node import WebNode
from core.rope import SpiderWebRoPE


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

        self.lm_head = nn.utils.spectral_norm(
            nn.Linear(self.d, config.model.vocab_size, bias=False)
        )
        self.final_norm = nn.RMSNorm(self.d)

    def forward(self, input_ids, tau=1.0, hard=False):
        B, T   = input_ids.shape
        device = input_ids.device
        NR     = self.cfg.model.num_rings
        NN     = self.cfg.model.nodes_per_ring

        # -------------------------
        # EMBEDDING + ROPE
        # -------------------------
        x = self.embed(input_ids)
        x = self.rope(x, node_idx=0, ring_idx=NR - 1)

        # -------------------------
        # MEMORY REGISTRY (B, slots, d)
        # -------------------------
        m_t_registry = [
            [self.rings[r][n].memory.init_memory(B, device)[0]
             for n in range(NN)]
            for r in range(NR)
        ]

        current_ring = torch.full((B, T), NR - 1, dtype=torch.long, device=device)
        active_mask  = torch.ones((B, T), dtype=torch.bool, device=device)

        all_aux_logits = []
        all_routings   = []
        spatial_chain  = [None] * NR

        # -------------------------
        # HOP LOOP
        # -------------------------
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

                    # -------------------------
                    # TOKEN GATHER
                    # -------------------------
                    flat_mask = node_mask.reshape(-1)
                    x_flat    = x.reshape(B * T, self.d)

                    h_in = x_flat[flat_mask]
                    num_active = h_in.size(0)

                    # -------------------------
                    # NEIGHBOURS
                    # -------------------------
                    ring_flat = ring_mask.reshape(-1)
                    ring_states = x_flat[ring_flat]

                    if ring_states.size(0) > 0:
                        mean = ring_states.mean(0, keepdim=True)
                        neighbours = mean.unsqueeze(1).expand(num_active, 1, self.d)
                    else:
                        neighbours = torch.zeros(num_active, 1, self.d, device=device)

                    # -------------------------
                    # 🔥 FIXED MEMORY (TOKEN-ALIGNED)
                    # -------------------------
                    m_t = m_t_registry[r_idx][n_idx]  # (B, slots, d)

                    m_t_exp = m_t.unsqueeze(1).expand(B, T, -1, -1)
                    m_t_flat = m_t_exp.reshape(B * T, m_t.size(1), self.d)

                    m_t_in = m_t_flat[flat_mask]  # (num_active, slots, d)

                    # -------------------------
                    # SPATIAL MEMORY
                    # -------------------------
                    m_s = None
                    if r_idx + 1 < NR and spatial_chain[r_idx + 1] is not None:
                        m_s_full = spatial_chain[r_idx + 1]

                        m_s_exp = m_s_full.unsqueeze(1).expand(B, T, -1, -1)
                        m_s_flat = m_s_exp.reshape(B * T, m_s_full.size(1), self.d)

                        m_s = m_s_flat[flat_mask]

                    # -------------------------
                    # NODE FORWARD
                    # -------------------------
                    out = node(h_in, neighbours, m_t_in, m_s, tau=tau, hard=hard)

                    # -------------------------
                    # WRITE BACK HIDDEN
                    # -------------------------
                    new_x_flat = new_x.reshape(B * T, self.d)
                    new_x_flat[flat_mask] = out["h"]
                    new_x = new_x_flat.reshape(B, T, self.d)

                    # -------------------------
                    # 🔥 WRITE BACK MEMORY
                    # -------------------------
                    m_t_flat_full = m_t_flat.clone()
                    m_t_flat_full[flat_mask] = out["m_t"].detach().to(m_t_flat_full.dtype)

                    m_t_new = m_t_flat_full.reshape(B, T, m_t.size(1), self.d).mean(dim=1)

                    m_t_registry[r_idx][n_idx] = m_t_new

                    # -------------------------
                    # ROUTING
                    # -------------------------
                    routing = out["routing"]
                    decisions = routing.argmax(dim=-1)

                    idxs = flat_mask.nonzero(as_tuple=True)[0]

                    cr_flat = current_ring.reshape(-1)
                    am_flat = active_mask.reshape(-1)

                    # move inward
                    move_in = (decisions == 1) & (cr_flat[idxs] > 0)
                    cr_flat[idxs[move_in]] -= 1

                    # exit
                    exit_mask = (decisions == 2) | (cr_flat[idxs] == 0)
                    am_flat[idxs[exit_mask]] = False

                    current_ring = cr_flat.reshape(B, T)
                    active_mask  = am_flat.reshape(B, T)

                    all_aux_logits.append(out["aux_logits"])
                    all_routings.append(routing)

                # -------------------------
                # SPATIAL CHAIN UPDATE
                # -------------------------
                ring_flat_all = ring_mask.reshape(-1)
                if ring_flat_all.any():
                    ring_h = new_x.reshape(B * T, self.d)[ring_flat_all]
                    slots  = self.cfg.memory.slots

                    mean = ring_h.mean(0)
                    spatial_chain[r_idx] = mean.unsqueeze(0).unsqueeze(0).expand(
                        B, slots, self.d
                    ).contiguous()

            x = new_x

            if not active_mask.any():
                break

        # -------------------------
        # FINAL
        # -------------------------
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return {
            "logits": logits,
            "aux_logits": all_aux_logits,
            "routings": all_routings,
        }