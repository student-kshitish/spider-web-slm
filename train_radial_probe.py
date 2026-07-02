"""
Step 3 — Radial hierarchy probe run.

Configuration: dim=64, fp32 master weights + bf16 autocast, seq_len=128.
Training run: exactly 2000 steps, fresh init (no checkpoint).
Weights: w_depth=0.02, w_recall=0.01 (low; won't destabilise CE).

Reports:
  - CE every 250 steps (stability check)
  - Ring-usage table at step 0 and step 2000
  - NaN / collapse check throughout
"""

import os
os.environ["WANDB_MODE"] = "disabled"

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import sentencepiece as spm

from config import (
    Config, ModelConfig, MemoryConfig, LorenzConfig, RoutingConfig, TrainConfig,
)
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.dataloader import get_dataloader
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

torch.manual_seed(42)

# ── probe config: dim=64, seq_len=128 (seq_len is max_seq_len in ModelConfig) ─

def probe_config() -> Config:
    return Config(
        model=ModelConfig(
            dim=64,
            hidden_dim=256,
            num_rings=4,
            nodes_per_ring=8,
            vocab_size=5000,
            max_seq_len=128,
        ),
        memory=MemoryConfig(slots=16, alpha=0.9, beta=0.1),
        lorenz=LorenzConfig(),
        routing=RoutingConfig(
            temp_start=2.0,
            temp_end=0.5,
            anneal_steps=2000,
            max_hops=6,
        ),
        train=TrainConfig(
            batch_size=16,
            lr=3e-4,
            lr_min=1e-5,
            weight_decay=0.01,
            grad_clip=1.0,
            warmup_steps=100,
            steps=2000,
            use_bf16=True,
            use_compile=False,
            entropy_weight=0.05,
            balance_weight=0.001,
        ),
    )


# ── radial hierarchy loss weights ─────────────────────────────────────────────
W_DEPTH  = 0.02   # outer rings: penalise exit over move_in
W_RECALL = 0.01   # inner rings: recall lagged token embedding


# ── ring stats helper ─────────────────────────────────────────────────────────

def print_ring_stats(ring_stats: dict, NR: int, label: str):
    total_visits = sum(ring_stats[r]["visits"] for r in range(NR)) or 1
    total_exits  = sum(ring_stats[r]["exit_count"] for r in range(NR)) or 1
    print(f"\n  Ring usage — {label}")
    print(f"  {'Ring':<14} {'Visits':>8} {'%Visit':>7} {'Stay':>7} {'In':>7} {'Exit':>7} {'ExCnt':>7} {'H-Norm':>8}")
    print(f"  {'-'*66}")
    for r in range(NR):
        rs  = ring_stats[r]
        rm  = rs["routing_mean"]
        tag = "[inner]" if r < NR // 2 else "[outer]"
        print(
            f"  Ring {r} {tag:<8}"
            f" {rs['visits']:>8,d}"
            f" {rs['visit_frac']*100:>6.1f}%"
            f"  {rm[0]:>5.3f}  {rm[1]:>5.3f}  {rm[2]:>5.3f}"
            f"  {rs['exit_count']:>6,d}"
            f"  {rs['h_norm_mean']:>7.3f}"
        )
    inner = sum(ring_stats[r]["visits"] for r in range(NR // 2)) / total_visits
    print(f"  >> Inner rings (0-{NR//2-1}): {inner*100:.1f}% of visits", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = probe_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NR     = cfg.model.num_rings

    print(f"Device : {device}")
    print(f"Config : dim={cfg.model.dim}, seq_len={cfg.model.max_seq_len}, "
          f"batch={cfg.train.batch_size}")
    print(f"Weights: w_depth={W_DEPTH}, w_recall={W_RECALL}")
    print()

    # ── model (FRESH init, no checkpoint) ────────────────────────────────────
    model = SpiderWeb(cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Params : {total_p/1e6:.2f}M (fresh init)\n")

    # ── loss / optimizer / schedulers ────────────────────────────────────────
    loss_fn   = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )
    lr_sched  = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)

    on_cuda      = device.type == "cuda"
    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16)
        if on_cuda and cfg.train.use_bf16
        else nullcontext()
    )

    # ── dataloader ───────────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")
    loader    = get_dataloader(cfg, sp)
    data_iter = iter(loader)

    # ── step-0 ring stats (before any training) ───────────────────────────────
    model.eval()
    with torch.no_grad():
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, _ = next(data_iter)
        x0  = x0.to(device)
        rs0 = model(x0, tau=1.0, hard=False, return_ring_stats=True)["ring_stats"]
    print_ring_stats(rs0, NR, "step 0 (before training)")

    # ── train loop (2000 steps) ───────────────────────────────────────────────
    model.train()
    ema_loss = None

    print(f"\n{'─'*60}")
    print(f"{'Step':>6}  {'CE':>8}  {'EMA-CE':>8}  {'Depth':>8}  {'Recall':>8}")
    print(f"{'─'*60}")

    for step in range(cfg.train.steps):

        # batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        tau  = tau_sched.get_temp(step)

        with autocast_ctx:
            out = model(x, tau=tau, hard=False)
            if cfg.train.use_bf16 and on_cuda:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(
                out, y,
                entropy_weight=cfg.train.entropy_weight,
                w_depth=W_DEPTH,
                w_recall=W_RECALL,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⛔  NaN/Inf at step {step}. Stopping.")
            return

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()
        lr_sched.step()

        ce = mets["ce"]
        ema_loss = ce if ema_loss is None else 0.95 * ema_loss + 0.05 * ce

        if step % 250 == 0 or step == cfg.train.steps - 1:
            print(
                f"{step:>6}  {ce:>8.4f}  {ema_loss:>8.4f}"
                f"  {mets['depth']:>8.4f}  {mets['recall']:>8.4f}",
                flush=True,
            )
            if step > 0 and (step % 500 == 0 or step == cfg.train.steps - 1):
                os.makedirs("checkpoints/radial_probe", exist_ok=True)
                torch.save(
                    {
                        "model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "step": step + 1,
                        "ema_ce": ema_loss,
                        "w_depth": W_DEPTH,
                        "w_recall": W_RECALL,
                    },
                    "checkpoints/radial_probe/last.pt",
                )
                print(f"  ↳ saved checkpoints/radial_probe/last.pt (step {step + 1})", flush=True)

    # ── step-2000 ring stats ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        try:
            xf, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xf, _ = next(data_iter)
        xf  = xf.to(device)
        rsf = model(xf, tau=0.5, hard=False, return_ring_stats=True)["ring_stats"]
    print_ring_stats(rsf, NR, "step 2000 (after training)")

    # ── comparison summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON  (inner ring visit fraction)")
    inner0 = sum(rs0[r]["visit_frac"] for r in range(NR // 2)) * 100
    innerf = sum(rsf[r]["visit_frac"] for r in range(NR // 2)) * 100
    print(f"  Step    0: {inner0:.1f}%")
    print(f"  Step 2000: {innerf:.1f}%")
    delta = innerf - inner0
    if delta > 5:
        print(f"  ✓ Inner rings gained +{delta:.1f} pp — mechanism is working")
    elif delta > 0:
        print(f"  ~ Slight improvement (+{delta:.1f} pp) — may need stronger weights")
    else:
        print(f"  ✗ No improvement ({delta:.1f} pp) — mechanism needs rework")
    print(f"{'='*60}")

    # ── save checkpoint ───────────────────────────────────────────────────────
    os.makedirs("checkpoints/radial_probe", exist_ok=True)
    ckpt_path = "checkpoints/radial_probe/best.pt"
    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "step": cfg.train.steps,
            "ema_ce": ema_loss,
            "w_depth": W_DEPTH,
            "w_recall": W_RECALL,
        },
        ckpt_path,
    )
    print(f"\n✓  Saved checkpoint → {ckpt_path}")

    # ── NaN check on final weights ────────────────────────────────────────────
    nan_params = [n for n, p in model.named_parameters() if p.isnan().any()]
    if nan_params:
        print(f"\n⛔  NaN in params: {nan_params}")
    else:
        print("\n✓  No NaN in any parameter (weights are healthy)")


if __name__ == "__main__":
    main()
