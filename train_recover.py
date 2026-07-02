"""
Recovery run: match OLD working run (CE 3.73 @ step 2709) exactly.

Three confirmed differences vs train_fixed.py, all restored here:
  1. weight_decay = 0.1  (was 0.001 — 100x gap)
  2. warmup_steps = 1203 (was 200 — reverse-engineered from LR@2709=0.00019872)
  3. entropy_weight follows old curriculum:
       steps  0-3000 : 0.1   (P1_Atomic_Logic)
       steps  3000-9000 : 0.05  (P2_Structural_Syntax)
       steps  9000-15000: 0.02  (P3_Stabilization)
       steps 15000+    : 0.01  (P4 — hard routing, but we stop at 3000)

Everything else matches the confirmed-identical column:
  lr=0.0002, betas=(0.9,0.95), eps=1e-8, grad_clip=1.0
  batch=16, seq_len=128, dim=64, hidden=256
  fp32 master weights + autocast(bfloat16) for compute
  hard_routing=False (soft throughout — old run was soft when it hit 3.73)

Stop at step 3000. Decisive test: does this run reach ~3.73 like the old one,
or does it stall ~6.6 like the slow fixed run?
"""

import os
os.environ["WANDB_MODE"] = "disabled"

import csv, math
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
TOTAL_STEPS   = 3000
WEIGHT_DECAY  = 0.1
LR            = 0.0002
WARMUP_STEPS  = 1203

# Entropy schedule matching old curriculum
def get_entropy_weight(step: int) -> float:
    if step < 3000:
        return 0.1    # P1_Atomic_Logic
    elif step < 9000:
        return 0.05   # P2_Structural_Syntax
    elif step < 15000:
        return 0.02   # P3_Stabilization
    else:
        return 0.01   # P4_Logical_Scaling

CKPT_DIR = Path("checkpoints/recover_run")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"
LOG_PATH  = Path("logs/recover_run.csv")

# ── Device ────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
on_cuda = device.type == "cuda"
if on_cuda:
    props = torch.cuda.get_device_properties(device)
    print(f"GPU : {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

# ── Model (dim=64, fp32 master weights) ──────────────────────────────────────
cfg = get_config()
cfg.model.dim        = 64
cfg.model.hidden_dim = 256
cfg.train.lr          = LR
cfg.train.weight_decay = WEIGHT_DECAY
cfg.train.warmup_steps = WARMUP_STEPS
cfg.train.steps        = TOTAL_STEPS   # shapes LR cosine tail correctly
cfg.train.lr_min       = 1e-5

model = SpiderWeb(cfg).to(device)
assert next(model.parameters()).dtype == torch.float32, "model must be float32"

total = sum(p.numel() for p in model.parameters())
print(f"Params : {total/1e6:.2f}M  dtype: {next(model.parameters()).dtype}")

# ── Norm snapshot helper ──────────────────────────────────────────────────────
def norm_snapshot(m):
    return {
        "final_norm.weight":            m.final_norm.weight.detach().clone(),
        "rings[3][0].attn_norm.weight": m.rings[3][0].attn_norm.weight.detach().clone(),
        "rings[1][0].attn_norm.weight": m.rings[1][0].attn_norm.weight.detach().clone(),
    }

snap0 = norm_snapshot(model)
print("Norms at step 0 (expect 1.0):")
for k, v in snap0.items():
    print(f"  {k}: {v.mean().item():.6f}")

# ── Loss / Optimizer / Schedulers ─────────────────────────────────────────────
loss_fn   = SpiderWebLoss().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR,
    weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), eps=1e-8, fused=on_cuda,
)
lr_scheduler  = get_cosine_warmup_scheduler(optimizer, cfg)
tau_scheduler = TemperatureScheduler(cfg)

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
csv_writer.writerow(["step", "ce", "ema", "lr", "tau", "ent_w"])

# ── Train ─────────────────────────────────────────────────────────────────────
best_loss = float("inf")
ema_loss  = None

print(f"\n{'='*65}")
print(f"RECOVERY RUN — matching old 3.73 config")
print(f"  wd={WEIGHT_DECAY}  warmup={WARMUP_STEPS}  entropy=curriculum  hard=False")
print(f"  dim=64  fp32-master+autocast-bf16  steps={TOTAL_STEPS}")
print(f"  checkpoints → {CKPT_DIR}")
print(f"{'='*65}\n")
print(f"  {'step':>6}  {'CE':>8}  {'EMA':>8}  {'lr':>12}  {'tau':>6}  {'ent_w':>6}")
print("  " + "-"*55)

model.train()
norm_check_done = False

for step in range(TOTAL_STEPS):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(loader)
        x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    tau       = tau_scheduler.get_temp(step)
    ent_w     = get_entropy_weight(step)

    with autocast_ctx:
        out = model(x, tau=tau, hard=False)
        out["logits"] = out["logits"].float()
        loss, mets = loss_fn(out, y, entropy_weight=ent_w)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n⛔  NaN/Inf at step {step} — stopping.")
        break

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    optimizer.step()
    lr_scheduler.step()

    ce = mets["ce"]
    ema_loss = ce if ema_loss is None else 0.95 * ema_loss + 0.05 * ce
    lr = optimizer.param_groups[0]["lr"]

    # Norm check at step 300
    if step == 300 and not norm_check_done:
        snap300 = norm_snapshot(model)
        print("\n  ── Norm movement @ step 300 ──")
        for k in snap0:
            delta = (snap300[k].float() - snap0[k].float()).abs().max().item()
            print(f"    {k}: max|Δ|={delta:.3e}  {'✓ MOVED' if delta > 1e-7 else '✗ FROZEN'}")
        print()
        norm_check_done = True

    csv_writer.writerow([step, f"{ce:.4f}", f"{ema_loss:.4f}",
                         f"{lr:.6f}", f"{tau:.4f}", f"{ent_w}"])
    if step % 100 == 0:
        csv_file.flush()

    if step % 250 == 0:
        print(f"  {step:6d}  {ce:8.4f}  {ema_loss:8.4f}  {lr:12.8f}  {tau:6.4f}  {ent_w}", flush=True)

    # Save every 500 steps
    if step % 500 == 0 and step > 0:
        torch.save({
            "step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "ema_loss": ema_loss, "best_loss": best_loss,
        }, LAST_CKPT)

    if ce < best_loss:
        best_loss = ce
        torch.save({"model": model.state_dict(),
                    "best_loss": best_loss, "step": step}, BEST_CKPT)

csv_file.flush(); csv_file.close()

print(f"\n{'='*65}")
print(f"Done at step {TOTAL_STEPS-1}.")
print(f"best_loss={best_loss:.4f}  (old run was 3.729 at step 2709)")
if best_loss < 4.5:
    print("✓  Below 4.5 — trajectory matches old run. Cause isolated.")
else:
    print("✗  Above 4.5 — still tracking slow path. Investigate batch_size or temp.")
print(f"{'='*65}")
