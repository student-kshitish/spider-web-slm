"""
Full corrected training run — dim=64, fresh from step 0.

Fix: model stays float32 (master weights). autocast(bfloat16) wraps
forward/loss only. This prevents AdamW updates from rounding to zero
in bf16 at magnitude 1.0 (the bug that froze all 97 RMSNorm weights).

Checkpoints saved to checkpoints/fixed_run/ — does NOT touch the old
broken checkpoints in checkpoints/.

Config:
  dim=64, hidden=256, lr=0.0002, weight_decay=0.001
  entropy_weight=0.003 constant, hard=False
  30000 steps, fresh random init
"""

import os
os.environ["WANDB_MODE"] = "disabled"

import csv
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn

from config import get_config
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.dataloader import get_dataloader
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler
import sentencepiece as spm

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True
torch.backends.cudnn.benchmark   = True

# ── Run config ────────────────────────────────────────────────────────────────
TOTAL_STEPS    = 30_000
ENTROPY_WEIGHT = 0.003
HARD_ROUTING   = False
LR             = 0.0002
WEIGHT_DECAY   = 0.001

CKPT_DIR = Path("checkpoints/fixed_run")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

LOG_PATH  = Path("logs/fixed_run.csv")

# ── Device ────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
on_cuda = device.type == "cuda"
print(f"Device: {device}")
if on_cuda:
    props = torch.cuda.get_device_properties(device)
    print(f"GPU   : {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

# ── Model (dim=64) ────────────────────────────────────────────────────────────
cfg = get_config()
cfg.model.dim        = 64
cfg.model.hidden_dim = 256
cfg.train.lr          = LR
cfg.train.weight_decay = WEIGHT_DECAY
cfg.train.steps        = TOTAL_STEPS

# CRITICAL: keep master weights in float32
model = SpiderWeb(cfg).to(device)
assert next(model.parameters()).dtype == torch.float32, "model must be float32"

total = sum(p.numel() for p in model.parameters())
print(f"Params: {total/1e6:.2f}M  dtype: {next(model.parameters()).dtype}")

# ── Norm snapshot helper ──────────────────────────────────────────────────────
def norm_snapshot(m):
    return {
        "final_norm.weight":            m.final_norm.weight.detach().clone(),
        "rings[3][0].attn_norm.weight": m.rings[3][0].attn_norm.weight.detach().clone(),
        "rings[1][0].attn_norm.weight": m.rings[1][0].attn_norm.weight.detach().clone(),
    }

snap0 = norm_snapshot(model)
print("\nNorm values at step 0 (should all be ≈ 1.0):")
for k, v in snap0.items():
    print(f"  {k}: mean={v.mean().item():.6f}")

# ── Loss / Optimizer / Schedulers ─────────────────────────────────────────────
loss_fn   = SpiderWebLoss().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR,
    weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), fused=on_cuda,
)
lr_scheduler  = get_cosine_warmup_scheduler(optimizer, cfg)
tau_scheduler = TemperatureScheduler(cfg)

# autocast for bf16 matmul speed — model weights stay float32
autocast_ctx = (
    torch.autocast("cuda", torch.bfloat16) if on_cuda else nullcontext()
)

# ── Data ──────────────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.Load("data/tokenizer.model")
loader     = get_dataloader(cfg, sp)
train_iter = iter(loader)

# ── CSV log ───────────────────────────────────────────────────────────────────
csv_file   = open(LOG_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["step", "ce", "ema", "lr"])

# ── Training loop ─────────────────────────────────────────────────────────────
best_loss = float("inf")
ema_loss  = None

print(f"\n{'='*65}")
print(f"FIXED RUN — dim=64, fp32 master weights, fresh from step 0")
print(f"  entropy_weight={ENTROPY_WEIGHT}  hard={HARD_ROUTING}  steps={TOTAL_STEPS}")
print(f"  checkpoints → {CKPT_DIR}")
print(f"{'='*65}\n")
print(f"  {'step':>6}  {'CE':>8}  {'EMA':>8}  {'lr':>12}")
print("  " + "-"*46)

model.train()
norm_check_done = False

for step in range(TOTAL_STEPS):
    # ── Batch ─────────────────────────────────────────────────────────────────
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(loader)
        x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    tau = tau_scheduler.get_temp(step)

    # ── Forward ───────────────────────────────────────────────────────────────
    with autocast_ctx:
        out = model(x, tau=tau, hard=HARD_ROUTING)
        out["logits"] = out["logits"].float()
        loss, mets = loss_fn(out, y, entropy_weight=ENTROPY_WEIGHT)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n⛔  NaN/Inf at step {step} — stopping.")
        break

    # ── Backward ──────────────────────────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    optimizer.step()
    lr_scheduler.step()

    # ── Metrics ───────────────────────────────────────────────────────────────
    ce = mets["ce"]
    ema_loss = ce if ema_loss is None else 0.95 * ema_loss + 0.05 * ce
    lr = optimizer.param_groups[0]["lr"]

    # ── Norm movement check at step 300 ───────────────────────────────────────
    if step == 300 and not norm_check_done:
        snap300 = norm_snapshot(model)
        print("\n  ── Norm movement check at step 300 (warmup ends ≈200) ──")
        all_moved = True
        for k in snap0:
            delta = (snap300[k].float() - snap0[k].float()).abs().max().item()
            status = "✓ MOVED" if delta > 1e-7 else "✗ FROZEN"
            if delta <= 1e-7:
                all_moved = False
            print(f"    {k}: max|Δ|={delta:.3e}  {status}")
        if all_moved:
            print("  ✓  All norms moved — fp32 fix is LIVE in this run.\n")
        else:
            print("  ⛔  Some norms still frozen — FIX NOT WORKING, investigate!\n")
        norm_check_done = True

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_writer.writerow([step, f"{ce:.4f}", f"{ema_loss:.4f}", f"{lr:.6f}"])
    if step % 100 == 0:
        csv_file.flush()

    # ── Console (every 500 steps) ─────────────────────────────────────────────
    if step % 500 == 0:
        print(f"  {step:6d}  {ce:8.4f}  {ema_loss:8.4f}  {lr:12.8f}", flush=True)

    # ── Save last.pt (full resumable state, every 500 steps) ─────────────────
    if step % 500 == 0 and step > 0:
        torch.save({
            "step":         step,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "ema_loss":     ema_loss,
            "best_loss":    best_loss,
        }, LAST_CKPT)

    # ── Save best.pt (model weights only, whenever CE improves) ──────────────
    if ce < best_loss:
        best_loss = ce
        torch.save({
            "model":     model.state_dict(),
            "best_loss": best_loss,
            "step":      step,
        }, BEST_CKPT)

csv_file.flush()
csv_file.close()

print(f"\n{'='*65}")
print(f"Training complete. best_loss={best_loss:.4f}")
print(f"Checkpoints in {CKPT_DIR}")
print(f"Log: {LOG_PATH}")
print(f"{'='*65}")
