# train_main.py

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
from train.curriculum import CurriculumManager

from train.scheduler import (
    get_cosine_warmup_scheduler,
    TemperatureScheduler,
)

# ============================================================
# GLOBALS
# ============================================================

torch.manual_seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ============================================================
# PATHS
# ============================================================

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# last.pt  — full state for crash/Ctrl-C resume (model+opt+sched+step)
# best.pt  — model weights only, for inference
LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TRAIN_LOG = LOG_DIR / "training.csv"


# ============================================================
# DEVICE
# ============================================================

def setup_device(cfg):

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training on CPU will be slow")
        return torch.device("cpu"), torch.float32

    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(device)

    print(f"GPU   : {props.name}")
    print(f"VRAM  : {props.total_memory / 1e9:.2f} GB")

    dtype = torch.bfloat16 if cfg.train.use_bf16 else torch.float32
    print(f"dtype : {dtype}")

    return device, dtype


# ============================================================
# PARAM COUNT
# ============================================================

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")


# ============================================================
# TRAIN
# ============================================================

def train(steps_override=None):

    cfg = get_config()
    cfg.train.grad_clip = 1.0

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device, dtype = setup_device(cfg)
    on_cuda = device.type == "cuda"

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    # Keep master weights in float32. autocast below handles bf16 compute.
    # Casting to bf16 froze all RMSNorm weights (init=1.0; update < bf16 epsilon).
    model = SpiderWeb(cfg).to(device)
    count_params(model)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss_fn = SpiderWebLoss().to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
        fused=on_cuda,
    )

    # --------------------------------------------------------
    # SCHEDULERS
    # --------------------------------------------------------

    lr_scheduler = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_scheduler = TemperatureScheduler(cfg)

    # --------------------------------------------------------
    # RESUME  (last.pt = full state)  OR  WARM-START  (best.pt = model only)
    # --------------------------------------------------------

    best_loss  = float("inf")
    ema_loss   = None
    start_step = 0
    resuming   = False

    if LAST_CKPT.exists():

        print(f"\nResuming from {LAST_CKPT} …\n")
        ckpt = torch.load(LAST_CKPT, map_location=device)

        # Old checkpoints may have bf16 weights; cast to float32 master weights.
        sd = {k: v.float() if v.is_floating_point() else v
              for k, v in ckpt["model"].items()}
        model.load_state_dict(sd)
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_step = ckpt["step"] + 1
        best_loss  = ckpt.get("best_loss", float("inf"))
        ema_loss   = ckpt.get("ema_loss",  None)
        resuming   = True

        print(f"  model / optimizer / scheduler restored")
        print(f"  resuming at step {start_step}\n")

    elif BEST_CKPT.exists():

        print(f"\nWarm-starting model from {BEST_CKPT} (fresh optimizer + scheduler) …\n")
        ckpt = torch.load(BEST_CKPT, map_location=device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = {k: v.float() if v.is_floating_point() else v
                  for k, v in ckpt["model"].items()}
            model.load_state_dict(sd)
            best_loss = ckpt.get("best_loss", float("inf"))
        else:
            sd = {k: v.float() if v.is_floating_point() else v
                  for k, v in ckpt.items()}
            model.load_state_dict(sd)

        print("  model weights loaded\n")

    # --------------------------------------------------------
    # AUTOCAST
    # --------------------------------------------------------

    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16)
        if on_cuda and cfg.train.use_bf16
        else nullcontext()
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    train_loader = get_dataloader(cfg, sp)
    train_iter   = iter(train_loader)

    # --------------------------------------------------------
    # CURRICULUM
    # --------------------------------------------------------

    curriculum = CurriculumManager()

    # --------------------------------------------------------
    # CSV LOG  (append on resume, create fresh otherwise)
    # --------------------------------------------------------

    csv_file   = open(TRAIN_LOG, "a" if resuming else "w", newline="")
    csv_writer = csv.writer(csv_file)

    if not resuming:
        csv_writer.writerow(["step", "ce", "ema", "lr", "phase"])

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------

    total_steps        = steps_override if steps_override is not None else cfg.train.steps
    console_log_every  = 500   # stdout: every 500 steps
    csv_log_every      = 50    # CSV:    every 50 steps

    consec_increase = 0
    prev_ema        = ema_loss

    print(f"Training for {total_steps} steps (start={start_step}) …\n")

    model.train()

    for step in range(start_step, total_steps):

        # ------------------------------------------------
        # BATCH
        # ------------------------------------------------

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        # ------------------------------------------------
        # CURRICULUM
        # ------------------------------------------------

        phase_cfg = curriculum.apply_to_model(step)
        hard      = phase_cfg["hard_routing"]

        # ------------------------------------------------
        # TEMPERATURE
        # ------------------------------------------------

        tau = tau_scheduler.get_temp(step)

        # ------------------------------------------------
        # FORWARD
        # ------------------------------------------------

        with autocast_ctx:

            out = model(x, tau=tau, hard=hard)

            if cfg.train.use_bf16:
                out["logits"] = out["logits"].float()

            loss, mets = loss_fn(
                out, y,
                entropy_weight = phase_cfg["entropy_weight"],
            )

        # ------------------------------------------------
        # NaN GUARD  — check before backward
        # ------------------------------------------------

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⛔  NaN/Inf loss at step {step} — stopping.")
            break

        # ------------------------------------------------
        # BACKWARD
        # ------------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        optimizer.step()
        lr_scheduler.step()

        # ------------------------------------------------
        # EMA LOSS
        # ------------------------------------------------

        current_loss = mets["ce"]

        if ema_loss is None:
            ema_loss = current_loss
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * current_loss

        # ------------------------------------------------
        # CONSECUTIVE INCREASE GUARD
        # ------------------------------------------------

        if prev_ema is not None and ema_loss > prev_ema:
            consec_increase += 1
        else:
            consec_increase = 0
        prev_ema = ema_loss

        if consec_increase > 1000:
            print(
                f"\n⛔  EMA loss rising for {consec_increase} consecutive steps "
                f"(step {step}) — stopping."
            )
            break

        # ------------------------------------------------
        # SAVE LAST  (full state — resumable)
        # ------------------------------------------------

        if step % 500 == 0 and step > start_step:
            torch.save(
                {
                    "step":         step,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "ema_loss":     ema_loss,
                    "best_loss":    best_loss,
                },
                LAST_CKPT,
            )

        # ------------------------------------------------
        # SAVE BEST  (model only — inference copy)
        # ------------------------------------------------

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(
                {
                    "model":     model.state_dict(),
                    "best_loss": best_loss,
                },
                BEST_CKPT,
            )

        # ------------------------------------------------
        # CSV LOG  (every 50 steps)
        # ------------------------------------------------

        if step % csv_log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                step,
                f"{current_loss:.4f}",
                f"{ema_loss:.4f}",
                f"{lr:.6f}",
                phase_cfg["name"],
            ])
            csv_file.flush()

        # ------------------------------------------------
        # CONSOLE LOG  (every 500 steps; every 100 in P4-transition window)
        # ------------------------------------------------

        in_p4_window = 14500 <= step <= 16000
        if step % console_log_every == 0 or (in_p4_window and step % 100 == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:6d} | "
                f"ce {current_loss:.4f} | "
                f"ema {ema_loss:.4f} | "
                f"lr {lr:.6f} | "
                f"{phase_cfg['name']}",
                flush=True,
            )

    print("\nTraining complete.")
    csv_file.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None,
                        help="Override cfg.train.steps (useful for sanity runs)")
    args = parser.parse_args()
    train(steps_override=args.steps)
