# train_ablation.py
# Chaos ablation: linear router replacing LorenzRouter.
# Everything else — SRM, RoPE, dims, training — identical to train_final.py.
#
# LinearRouter keeps:  proj_in, proj_out, eps, Jacobian guard, memory integration,
#                      spectral_norm, Gumbel-softmax (identical to LorenzRouter)
# LinearRouter drops:  the 5-step Lorenz ODE (_lorenz_step loop)
# Param count per node: 202 — exact match with LorenzRouter.

import os
os.environ["WANDB_MODE"] = "disabled"

import csv
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from config import get_config
from core.web import SpiderWeb
from core.node import WebNode
from core.rope import SpiderWebRoPE
from train.loss import SpiderWebLoss
from train.dataloader import get_dataloader
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

# ============================================================
# LINEAR ROUTER  (LorenzRouter with ODE loop removed)
# ============================================================

class LinearRouter(nn.Module):
    """
    Ablation baseline for LorenzRouter.
    Identical structure (proj_in, proj_out, eps, Jacobian guard, memory blend,
    spectral_norm, Gumbel-softmax) but replaces the 5-step Lorenz ODE with
    a direct linear pass.  Trainable params per node: 202 (exact match).
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.model.dim
        self.eps_init = config.lorenz.eps_init

        # Identical projections to LorenzRouter — spectral_norm kept for fairness
        self.proj_in  = spectral_norm(nn.Linear(self.dim, 3, bias=False))   # 192 params
        self.proj_out = spectral_norm(nn.Linear(3, 3, bias=False))           # 9 params
        self.eps      = nn.Parameter(torch.tensor([self.eps_init]))          # 1 param
        # Total: 202 — same as LorenzRouter

    def forward(self, h, M_t, tau=1.0, hard=False):
        # --- Jacobian guard (identical to LorenzRouter) ---
        xyz0 = self.proj_in(h)
        f_h  = self.proj_out(xyz0)
        h_guarded = h + self.eps * torch.tanh(
            F.linear(f_h, self.proj_in.weight.t())
        )

        # --- Memory integration (identical to LorenzRouter) ---
        mem_ctx = M_t.mean(dim=1)
        hidden  = self.proj_in(h_guarded + 0.1 * mem_ctx)

        # --- Linear routing decision (ODE loop removed) ---
        logits  = self.proj_out(hidden)

        # --- Gumbel-softmax (identical) ---
        routing = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        return routing, h_guarded


# ============================================================
# WEBNODE WITH LINEAR ROUTER
# ============================================================

class WebNodeLinear(WebNode):
    """WebNode with router swapped to LinearRouter. Everything else unchanged."""

    def __init__(self, config):
        super().__init__(config)
        self.router = LinearRouter(config)   # overwrite LorenzRouter


# ============================================================
# SPIDERWEB WITH LINEAR ROUTER
# ============================================================

class SpiderWebLinear(SpiderWeb):
    """SpiderWeb using WebNodeLinear. forward() inherited unchanged."""

    def __init__(self, config):
        # Bypass SpiderWeb.__init__ to control node type;
        # re-build all identical attributes.
        nn.Module.__init__(self)
        self.cfg = config
        self.d   = config.model.dim

        self.embed = nn.Embedding(config.model.vocab_size, self.d)
        self.rope  = SpiderWebRoPE(config)

        self.rings = nn.ModuleList([
            nn.ModuleList([
                WebNodeLinear(config) for _ in range(config.model.nodes_per_ring)
            ])
            for _ in range(config.model.num_rings)
        ])

        self.lm_head    = spectral_norm(nn.Linear(self.d, config.model.vocab_size, bias=False))
        self.final_norm = nn.RMSNorm(self.d)


# ============================================================
# CONSTANTS  (identical to train_final.py)
# ============================================================

TOTAL_STEPS    = 20000
ENTROPY_WEIGHT = 0.003
HARD_ROUTING   = False

CKPT_DIR = Path("checkpoints/ablation_linear")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

LOG_FILE = Path("logs/ablation_linear.csv")

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# ============================================================
# PARAM COMPARISON
# ============================================================

def param_report(chaos_model, linear_model):
    def count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    c_total = count(chaos_model)
    l_total = count(linear_model)

    # router params per node
    c_router = count(chaos_model.rings[0][0].router)
    l_router = count(linear_model.rings[0][0].router)
    n_nodes  = chaos_model.cfg.model.num_rings * chaos_model.cfg.model.nodes_per_ring

    print("Param comparison:")
    print(f"  Chaos router per node  : {c_router:,}  ×{n_nodes} nodes = {c_router*n_nodes:,}")
    print(f"  Linear router per node : {l_router:,}  ×{n_nodes} nodes = {l_router*n_nodes:,}")
    print(f"  Chaos model total      : {c_total/1e6:.4f}M")
    print(f"  Linear model total     : {l_total/1e6:.4f}M")
    print(f"  Difference             : {abs(c_total-l_total):,} params")
    print()


# ============================================================
# TRAIN
# ============================================================

def train():

    # --------------------------------------------------------
    # CONFIG  (identical to train_final.py)
    # --------------------------------------------------------
    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
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

    print(f"\nAblation run config (identical to chaos run):")
    print(f"  dim=64  hidden=256  lr=0.0002  wd=0.001")
    print(f"  betas=(0.9,0.95)  grad_clip=1.0  batch=16  seq_len=128")
    print(f"  entropy_weight=0.003 constant  hard_routing=False")
    print(f"  fp32 master weights + autocast(bfloat16)")
    print(f"  steps=20000  fresh from step 0")
    print(f"  router: LinearRouter (Lorenz ODE removed)")
    print(f"  save → {CKPT_DIR}/\n")

    # --------------------------------------------------------
    # MODELS (both, for param comparison only)
    # --------------------------------------------------------
    chaos_model  = SpiderWeb(cfg).to(device)
    linear_model = SpiderWebLinear(cfg).to(device)

    param_report(chaos_model, linear_model)

    # Free chaos model — not training it here
    del chaos_model
    torch.cuda.empty_cache() if on_cuda else None

    model = linear_model

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
    # AUTOCAST
    # --------------------------------------------------------
    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16) if on_cuda else nullcontext()
    )

    # --------------------------------------------------------
    # STEP-0 CE CHECK
    # --------------------------------------------------------
    model.eval()
    x0, y0 = next(train_iter)
    x0, y0 = x0.to(device), y0.to(device)
    with torch.no_grad(), autocast_ctx:
        out0 = model(x0, tau=tau_scheduler.get_temp(0), hard=False)
        out0["logits"] = out0["logits"].float()
        _, m0 = loss_fn(out0, y0, entropy_weight=ENTROPY_WEIGHT)
    print(f"[SANITY] step=0 random-init CE = {m0['ce']:.4f}  (expected ~8.52)")

    norm_w0 = model.final_norm.weight.detach().cpu().clone()
    print(f"[SANITY] final_norm.weight init mean = {norm_w0.mean().item():.6f}")
    print()

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
    ce              = m0["ce"]

    print(f"Training linear-router ablation — {TOTAL_STEPS} steps from step 0.\n")

    for step in range(TOTAL_STEPS):

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        tau = tau_scheduler.get_temp(step)

        with autocast_ctx:
            out = model(x, tau=tau, hard=HARD_ROUTING)
            out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=ENTROPY_WEIGHT)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⛔  NaN/Inf at step {step} — stopping.")
            break

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

        if prev_ema is not None and ema_loss > prev_ema:
            consec_increase += 1
        else:
            consec_increase = 0
        prev_ema = ema_loss

        if consec_increase > 1000:
            print(f"\n⛔  EMA rising {consec_increase} consecutive steps at step {step} — stopping.")
            break

        if step == 300:
            norm_w300 = model.final_norm.weight.detach().cpu()
            delta_pct = (norm_w300 - norm_w0).abs().mean().item() / norm_w0.abs().mean().item() * 100
            status    = "✓ OK" if delta_pct > 0.1 else "⛔ FROZEN"
            print(f"[NORM CHECK] step=300  final_norm.weight moved {delta_pct:.2f}%  {status}", flush=True)
            print()

        if step % 500 == 0 and step > 0:
            torch.save({
                "step":         step,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "ema_loss":     ema_loss,
                "best_loss":    best_loss,
            }, LAST_CKPT)

        if ce < best_loss:
            best_loss = ce
            torch.save({"model": model.state_dict(), "best_loss": best_loss}, BEST_CKPT)

        if step % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            csv_writer.writerow([step, f"{ce:.4f}", f"{ema_loss:.4f}", f"{lr_now:.6f}"])
            csv_file.flush()

        if step % 2000 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"STEP {step:6d}/{TOTAL_STEPS} | CE {ce:.4f} | EMA {ema_loss:.4f} | LR {lr_now:.6f}",
                flush=True,
            )

    # --------------------------------------------------------
    # FINAL CHECKPOINT
    # --------------------------------------------------------
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
    print(f"Ablation training complete.")
    print(f"  Final step CE : {ce:.4f}")
    print(f"  Final EMA     : {ema_loss:.4f}")
    print(f"  Best CE seen  : {best_loss:.4f}")
    print(f"{'='*60}\n")

    # ========================================================
    # EVAL — same 50 batches as chaos model (seed=0, same loader)
    # ========================================================
    print("Running 50-batch eval (matching chaos model eval exactly) …\n")

    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    eval_loss_fn = SpiderWebLoss().to(device)
    torch.manual_seed(0)   # same seed as eval_final.py Test 2
    eval_loader = get_dataloader(cfg, sp)

    ces = []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            if i >= 50:
                break
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                out = model(x, tau=0.1, hard=False)
                out["logits"] = out["logits"].float()
                _, mets = eval_loss_fn(out, y, entropy_weight=ENTROPY_WEIGHT)
            ces.append(mets["ce"])

    avg_ce = sum(ces) / len(ces)
    min_ce = min(ces)

    # Norm movement
    norm_w_final = model.final_norm.weight.detach().cpu()
    norm_mean    = norm_w_final.mean().item()
    norm_pct     = (norm_w_final - 1.0).abs().mean().item() / 1.0 * 100

    # ========================================================
    # COMPARISON TABLE
    # ========================================================
    print("=" * 60)
    print("ABLATION RESULT — Chaos vs Linear Router")
    print("=" * 60)
    print(f"{'Metric':<30} {'Chaos':>10} {'Linear':>10}")
    print("-" * 52)
    print(f"{'Steps trained':<30} {'20000':>10} {'20000':>10}")
    print(f"{'Avg CE (50 batches)':<30} {'4.7677':>10} {avg_ce:>10.4f}")
    print(f"{'Min CE (50 batches)':<30} {'4.5328':>10} {min_ce:>10.4f}")
    print(f"{'Best ckpt CE':<30} {'4.4212':>10} {best_loss:>10.4f}")
    print(f"{'Final EMA':<30} {'4.7508':>10} {ema_loss:>10.4f}")
    print(f"{'final_norm moved from 1.0':<30} {'274.47%':>10} {norm_pct:>9.2f}%")
    print("-" * 52)
    delta = avg_ce - 4.7677
    direction = "worse" if delta > 0 else "better"
    print(f"Linear router avg CE is {abs(delta):.4f} {direction} than chaos router.")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()
