import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config

cfg = get_config()


class SpiderWebLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        model_output,
        targets,
        entropy_weight=None,
        w_depth=0.0,
        w_recall=0.0,
    ):
        logits        = model_output["logits"]      # (B, T, V)
        routings_list = model_output["routings"]

        B, T, V = logits.shape
        device  = logits.device

        if entropy_weight is None:
            entropy_weight = cfg.train.entropy_weight

        # ── 1. PRIMARY LOSS ───────────────────────────────────────────────────
        # Pointer/copy arm: the model returns a normalized mixture probability
        # P = lambda*P_gen + (1-lambda)*P_copy (web.py). P_copy is a probability,
        # not a logit, so we take NLL on log(P) instead of CE on raw logits. When
        # the pointer is OFF, "pointer" is None and this is the usual CE on logits.
        pointer = model_output.get("pointer", None)
        if pointer is not None:
            P     = pointer["P"]                                  # (B,T,V), sums to 1
            logP  = torch.log(P.clamp_min(1e-9))
            loss_ce = F.nll_loss(logP.view(-1, V), targets.view(-1),
                                 ignore_index=-1)
        else:
            loss_ce = self.ce(logits.view(-1, V), targets.view(-1))

        # ── 2. ROUTING ENTROPY ────────────────────────────────────────────────
        loss_ent      = torch.tensor(0.0, device=device)
        ent_count     = 0
        all_routing_means = []

        for r in routings_list:
            if isinstance(r, torch.Tensor):
                p        = r.clamp(1e-8, 1.0)
                node_ent = -(p * torch.log(p)).sum(-1).mean()
                loss_ent -= node_ent
                all_routing_means.append(r.mean(0))
                ent_count += 1

        loss_ent = loss_ent / max(ent_count, 1)

        # ── 3. LOAD BALANCE ───────────────────────────────────────────────────
        loss_bal = torch.tensor(0.0, device=device)

        if all_routing_means:
            stacked  = torch.stack(all_routing_means, dim=0)
            loss_bal = stacked.var(dim=0).mean()

        # ── 4. RADIAL HIERARCHY LOSSES ────────────────────────────────────────
        # (a) depth pressure: outer rings pay for choosing exit over move_in
        loss_depth = model_output.get(
            "depth_loss", torch.tensor(0.0, device=device)
        )

        # (b) inner-ring recall: memory must reconstruct lagged token embedding
        loss_recall = model_output.get(
            "recall_loss", torch.tensor(0.0, device=device)
        )

        # ── TOTAL LOSS ────────────────────────────────────────────────────────
        total = (
            loss_ce
            + entropy_weight        * loss_ent
            + cfg.train.balance_weight * loss_bal
            + w_depth               * loss_depth
            + w_recall              * loss_recall
        )

        return total, {
            "total":   total.item(),
            "ce":      loss_ce.item(),
            "ent":     loss_ent.item(),
            "bal":     loss_bal.item(),
            "depth":   loss_depth.item(),
            "recall":  loss_recall.item(),
        }
