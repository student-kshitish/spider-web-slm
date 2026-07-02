# train_finetune.py
# Warm-start fine-tune from checkpoints/final_run/best.pt (dim=64, CE ~4.77)
# on the larger dataset (data/raw/ now contains ~170MB).
# Lower LR: 1e-4, short warmup 100 steps, 6000 steps total.
# Saves to checkpoints/finetune_run/

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

# ============================================================
# CONSTANTS
# ============================================================

TOTAL_STEPS    = 6000
ENTROPY_WEIGHT = 0.003
HARD_ROUTING   = False

SOURCE_CKPT = Path("checkpoints/final_run/best.pt")
CKPT_DIR    = Path("checkpoints/finetune_run")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

LOG_FILE = Path("logs/finetune_run.csv")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# ============================================================
# TRAIN
# ============================================================

def train():

    # --------------------------------------------------------
    # CONFIG — dim=64 to match the source checkpoint
    # --------------------------------------------------------
    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = 128
    cfg.train.lr          = 1e-4        # half the original 2e-4
    cfg.train.lr_min      = 1e-5
    cfg.train.weight_decay = 0.001
    cfg.train.grad_clip   = 1.0
    cfg.train.warmup_steps = 100        # short warmup — already trained
    cfg.train.steps       = TOTAL_STEPS
    cfg.train.batch_size  = 16
    cfg.train.use_bf16    = True
    cfg.train.entropy_weight = ENTROPY_WEIGHT

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"

    if on_cuda:
        props = torch.cuda.get_device_properties(device)
        print(f"GPU  : {props.name}")
        print(f"VRAM : {props.total_memory / 1e9:.2f} GB")

    print(f"\nFine-tune config:")
    print(f"  dim=64  hidden=256  lr=1e-4 (half of original)  wd=0.001")
    print(f"  warmup=100 steps  batch=16  seq_len=128")
    print(f"  entropy_weight=0.003 constant  hard_routing=False")
    print(f"  fp32 master weights + autocast(bfloat16)")
    print(f"  steps={TOTAL_STEPS}  warm start from {SOURCE_CKPT}")
    print(f"  save → {CKPT_DIR}/\n")

    # --------------------------------------------------------
    # MODEL  (fp32 master weights)
    # --------------------------------------------------------
    model = SpiderWeb(cfg).to(device)

    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, f"Expected fp32, got: {dtypes}"

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total/1e6:.3f}M")

    # --------------------------------------------------------
    # WARM START — load source checkpoint
    # --------------------------------------------------------
    print(f"Loading {SOURCE_CKPT} ...")
    src = torch.load(SOURCE_CKPT, map_location=device)
    src_state = src["model"] if "model" in src else src
    missing, unexpected = model.load_state_dict(src_state, strict=True)
    if missing:
        print(f"  WARNING missing keys: {missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    print(f"  Source best_loss: {src.get('best_loss', 'N/A')}")
    print(f"  Weights loaded ✓\n")

    # --------------------------------------------------------
    # LOSS / OPTIMIZER / SCHEDULERS
    # --------------------------------------------------------
    loss_fn = SpiderWebLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
        fused=on_cuda,
    )

    lr_scheduler  = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_scheduler = TemperatureScheduler(cfg)

    # --------------------------------------------------------
    # TOKENIZER + DATALOADER (picks up all .txt in data/raw/)
    # --------------------------------------------------------
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")

    train_loader = get_dataloader(cfg, sp)
    train_iter   = iter(train_loader)

    import glob, os
    txt_files = glob.glob("data/raw/*.txt")
    total_mb  = sum(os.path.getsize(f) for f in txt_files) / 1e6
    print(f"Data: {len(txt_files)} file(s), {total_mb:.1f} MB total")
    for f in txt_files:
        print(f"  {f}  ({os.path.getsize(f)/1e6:.1f} MB)")
    print()

    # --------------------------------------------------------
    # AUTOCAST
    # --------------------------------------------------------
    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16) if on_cuda else nullcontext()
    )

    # ========================================================
    # SANITY — warm-start CE check (expect ~4.7, NOT ~8.5)
    # ========================================================
    model.train()
    x0, y0 = next(train_iter)
    x0, y0 = x0.to(device), y0.to(device)
    with torch.no_grad(), autocast_ctx:
        out0 = model(x0, tau=tau_scheduler.get_temp(0), hard=False)
        out0["logits"] = out0["logits"].float()
        _, m0 = loss_fn(out0, y0, entropy_weight=ENTROPY_WEIGHT)
    init_ce = m0["ce"]

    print(f"[SANITY] Warm-start CE = {init_ce:.4f}  (expect ~4.7, abort if ~8.5)")

    if init_ce > 7.0:
        print(f"\n⛔  ABORT: CE={init_ce:.4f} — warm start failed (looks like fresh init).")
        print(f"  Check that {SOURCE_CKPT} exists and dim=64 matches the checkpoint.")
        return

    print(f"  Warm start confirmed ✓\n")

    # Reset iterator so training starts from the beginning
    train_iter = iter(train_loader)

    model.train()

    # --------------------------------------------------------
    # CSV LOG
    # --------------------------------------------------------
    csv_file   = open(LOG_FILE, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "ce", "ema", "lr"])

    # ========================================================
    # TRAINING LOOP
    # ========================================================
    best_loss       = init_ce   # start tracking from warm-start level
    ema_loss        = None
    prev_ema        = None
    consec_increase = 0
    ce              = init_ce

    print(f"Fine-tuning — {TOTAL_STEPS} steps from warm start (CE={init_ce:.4f}).\n")

    for step in range(TOTAL_STEPS):

        # ---- batch ----
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        tau = tau_scheduler.get_temp(step)

        # ---- forward ----
        with autocast_ctx:
            out = model(x, tau=tau, hard=HARD_ROUTING)
            out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=ENTROPY_WEIGHT)

        # ---- NaN guard ----
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⛔  NaN/Inf at step {step} — stopping.")
            break

        # ---- backward ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()
        lr_scheduler.step()

        ce = mets["ce"]
        ema_loss = ce if ema_loss is None else 0.95 * ema_loss + 0.05 * ce

        # ---- consecutive-increase guard ----
        if prev_ema is not None and ema_loss > prev_ema:
            consec_increase += 1
        else:
            consec_increase = 0
        prev_ema = ema_loss

        if consec_increase > 1000:
            print(f"\n⛔  EMA rising {consec_increase} steps at step {step} — stopping.")
            break

        # ---- checkpoint: last (every 500) ----
        if step % 500 == 0 and step > 0:
            torch.save({
                "step":         step,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "ema_loss":     ema_loss,
                "best_loss":    best_loss,
            }, LAST_CKPT)

        # ---- checkpoint: best ----
        if ce < best_loss:
            best_loss = ce
            torch.save({"model": model.state_dict(), "best_loss": best_loss}, BEST_CKPT)

        # ---- CSV every 50 steps ----
        if step % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            csv_writer.writerow([step, f"{ce:.4f}", f"{ema_loss:.4f}", f"{lr_now:.6f}"])
            csv_file.flush()

        # ---- milestone print every 1000 steps ----
        if step % 1000 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"STEP {step:5d}/{TOTAL_STEPS} | CE {ce:.4f} | EMA {ema_loss:.4f} | LR {lr_now:.6f}",
                flush=True,
            )

    # ========================================================
    # FINAL CHECKPOINT + SUMMARY
    # ========================================================
    lr_now = optimizer.param_groups[0]["lr"]
    torch.save({
        "step":         TOTAL_STEPS - 1,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ema_loss":     ema_loss,
        "best_loss":    best_loss,
    }, LAST_CKPT)
    csv_file.close()

    print(f"\n{'='*60}")
    print(f"Fine-tune complete.")
    print(f"  Warm-start CE : {init_ce:.4f}")
    print(f"  Final step CE : {ce:.4f}")
    print(f"  Final EMA     : {ema_loss:.4f}")
    print(f"  Best CE seen  : {best_loss:.4f}")
    print(f"  Checkpoint    : {LAST_CKPT}")
    print(f"{'='*60}\n")

    # ========================================================
    # GENERATION SAMPLES
    # ========================================================
    import torch.nn.functional as F

    print("Loading best.pt for generation samples …\n")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [
        "Once upon a time",
        "The little dog",
        "One day a girl",
        "Tom and his friend",
    ]

    print("─" * 60)
    for prompt in prompts:
        ids = sp.EncodeAsIds(prompt)
        generated = list(ids)
        with torch.no_grad():
            for _ in range(80):
                x_in = torch.tensor([generated[-128:]], device=device)
                with autocast_ctx:
                    out = model(x_in, tau=0.8, hard=False)
                logits = out["logits"][0, -1].float()
                top_k  = 40
                vals, idxs = torch.topk(logits, top_k)
                probs  = torch.softmax(vals, dim=-1)
                next_id = idxs[torch.multinomial(probs, 1)].item()
                generated.append(next_id)
        text = sp.DecodeIds(generated)
        print(f"  [{prompt!r}]")
        print(f"  {text}")
        print()
    print("─" * 60)
    print("\nDone.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train()
