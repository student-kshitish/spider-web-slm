# train_scale.py
# Scale experiment: dim=96, hidden=384, fp32 master + autocast(bf16)
# 20000 steps from step 0, checkpoints/scale_run/
# NEVER overwrites checkpoints/final_run/

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

TOTAL_STEPS    = 20000
ENTROPY_WEIGHT = 0.003   # constant — no curriculum override
HARD_ROUTING   = False

CKPT_DIR = Path("checkpoints/scale_run")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

LOG_FILE = Path("logs/scale_run.csv")
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
    # CONFIG OVERRIDES
    # --------------------------------------------------------
    cfg = get_config()
    cfg.model.dim         = 96
    cfg.model.hidden_dim  = 384
    cfg.model.max_seq_len = 128
    cfg.train.lr          = 0.0002
    cfg.train.lr_min      = 1e-5
    cfg.train.weight_decay = 0.001
    cfg.train.grad_clip   = 1.0
    cfg.train.warmup_steps = 200
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
    else:
        print("WARNING: no CUDA — training on CPU")

    print(f"\nScale run config:")
    print(f"  dim=96  hidden=384  lr=0.0002  wd=0.001")
    print(f"  betas=(0.9,0.95)  grad_clip=1.0  batch={cfg.train.batch_size}  seq_len=128")
    print(f"  entropy_weight=0.003 constant  hard_routing=False")
    print(f"  fp32 master weights + autocast(bfloat16)")
    print(f"  steps={TOTAL_STEPS}  fresh from step 0")
    print(f"  save → {CKPT_DIR}/\n")

    # --------------------------------------------------------
    # MODEL  (float32 master weights — DO NOT call .to(bfloat16))
    # --------------------------------------------------------
    model = SpiderWeb(cfg).to(device)

    # Verify all parameters are fp32
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, f"Expected fp32 master weights, got: {dtypes}"

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable/1e6:.3f}M trainable / {total/1e6:.3f}M total")
    print(f"  (expected ~7M)\n")

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
    # TOKENIZER + DATALOADER
    # --------------------------------------------------------
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")

    train_loader = get_dataloader(cfg, sp)
    train_iter   = iter(train_loader)

    # --------------------------------------------------------
    # AUTOCAST  (compute in bf16, master weights stay fp32)
    # --------------------------------------------------------
    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16) if on_cuda else nullcontext()
    )

    # ========================================================
    # OOM PROBE  — try a dummy forward at batch=16 first
    # ========================================================
    if on_cuda:
        try:
            dummy_x = torch.zeros(cfg.train.batch_size, cfg.model.max_seq_len,
                                  dtype=torch.long, device=device)
            with torch.no_grad(), autocast_ctx:
                model(dummy_x, tau=1.0, hard=False)
            torch.cuda.synchronize()
            del dummy_x
            torch.cuda.empty_cache()
            print(f"OOM probe passed at batch={cfg.train.batch_size}.\n")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cfg.train.batch_size = 8
            train_loader = get_dataloader(cfg, sp)
            train_iter   = iter(train_loader)
            print(f"⚠  OOM at batch=16 — dropped to batch=8 and rebuilt dataloader.\n")

    # ========================================================
    # SANITY 1 — step-0 CE with random init (expected ~8.52)
    # Must use train mode: at dim=96 the lm_head uses nn.utils.spectral_norm
    # (old API) which needs one training-mode forward to run its power-iteration
    # step and get an accurate sigma estimate. Eval mode before any training
    # forward gives unconverged sigma from random u/v -> logits blow up -> CE ~22.
    # ========================================================
    model.train()
    x0, y0 = next(train_iter)
    x0, y0 = x0.to(device), y0.to(device)
    with torch.no_grad(), autocast_ctx:
        out0 = model(x0, tau=tau_scheduler.get_temp(0), hard=False)
        out0["logits"] = out0["logits"].float()
        _, m0 = loss_fn(out0, y0, entropy_weight=ENTROPY_WEIGHT)
    init_ce = m0["ce"]
    print(f"[SANITY 1] step=0 random-init CE = {init_ce:.4f}  (expected ~8.52)")

    # SANITY 2 — norm weights at init (expected 1.0)
    norm_w0 = model.final_norm.weight.detach().cpu().clone()
    print(f"[SANITY 2] final_norm.weight at step 0: mean={norm_w0.mean().item():.6f}  (expected 1.000000)")
    print()

    # Reset iterator so training starts fresh
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
    best_loss       = float("inf")
    ema_loss        = None
    prev_ema        = None
    consec_increase = 0
    ce              = init_ce

    print(f"Training — {TOTAL_STEPS} steps from step 0.\n")

    for step in range(TOTAL_STEPS):

        # ---- batch ----
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        tau = tau_scheduler.get_temp(step)

        # ---- forward (bf16 compute, fp32 master weights) ----
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
            print(f"\n⛔  EMA rising {consec_increase} consecutive steps at step {step} — stopping.")
            break

        # ---- norm-movement check at step 300 ----
        if step == 300:
            norm_w300 = model.final_norm.weight.detach().cpu()
            delta_pct = (norm_w300 - norm_w0).abs().mean().item() / norm_w0.abs().mean().item() * 100
            status    = "✓ OK" if delta_pct > 0.1 else "⛔ FROZEN"
            print(f"[NORM CHECK] step=300  final_norm.weight moved {delta_pct:.2f}% "
                  f"(mean={norm_w300.mean().item():.6f})  {status}", flush=True)
            print()

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

        # ---- milestone print every 2000 steps ----
        if step % 2000 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"STEP {step:6d}/{TOTAL_STEPS} | CE {ce:.4f} | EMA {ema_loss:.4f} | LR {lr_now:.6f}",
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
    print(f"Scale run complete.")
    print(f"  Final step CE : {ce:.4f}")
    print(f"  Final EMA     : {ema_loss:.4f}")
    print(f"  Best CE seen  : {best_loss:.4f}")
    print(f"  Checkpoint    : {LAST_CKPT}")
    print(f"{'='*60}\n")

    # ========================================================
    # GENERATION SAMPLES  (from best.pt)
    # ========================================================
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
