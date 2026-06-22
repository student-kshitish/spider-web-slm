import torch
import torch.nn as nn
from config import get_config

cfg = get_config()


class SpiderWebLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, model_output, targets, entropy_weight=None):
        logits        = model_output["logits"]      # (B, T, V)
        routings_list = model_output["routings"]

        B, T, V = logits.shape
        device = logits.device

        if entropy_weight is None:
            entropy_weight = cfg.train.entropy_weight

        # -------------------------
        # 1. PRIMARY LOSS
        # -------------------------
        loss_ce = self.ce(logits.view(-1, V), targets.view(-1))

        # -------------------------
        # 2. ROUTING ENTROPY
        # -------------------------
        loss_ent = torch.tensor(0.0, device=device)
        ent_count = 0
        all_routing_means = []

        for r in routings_list:
            if isinstance(r, torch.Tensor):
                p = r.clamp(1e-8, 1.0)

                node_ent = -(p * torch.log(p)).sum(-1).mean()
                loss_ent -= node_ent

                all_routing_means.append(r.mean(0))
                ent_count += 1

        loss_ent = loss_ent / max(ent_count, 1)

        # -------------------------
        # 4. LOAD BALANCE
        # -------------------------
        loss_bal = torch.tensor(0.0, device=device)

        if all_routing_means:
            stacked = torch.stack(all_routing_means, dim=0)
            loss_bal = stacked.var(dim=0).mean()

        # -------------------------
        # TOTAL LOSS
        # -------------------------
        total = (
            loss_ce
            + entropy_weight * loss_ent
            + cfg.train.balance_weight * loss_bal
        )

        return total, {
            "total": total.item(),
            "ce":    loss_ce.item(),
            "ent":   loss_ent.item(),
            "bal":   loss_bal.item(),
        }