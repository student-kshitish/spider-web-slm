"""
Step 2 — Train the inner-ring mechanism on dependency-rich data.

Config:
  dim=64, hidden_dim=256, fp32 master weights + bf16 autocast.
  Ring 0 fix active (confirmed in core/web.py — exit only on action 2).
  Variant A memory = SolarRingMemory (the single wired-in SRM v2.1).
  w_depth=0.005, w_recall=0.02  (tuned weights from the 2000-step probe).
  seq_len=256, batch=16 (auto-falls back to 8 then 4 on CUDA OOM).
  lr=2e-4, weight_decay=1e-3, fresh from step 0.
  ~8000 steps.

Data: 50/50 MIX of synthetic binding data (data/raw/binding.txt) and
TinyStories (data/raw/tinystories.txt) so it still learns general language.

Checks printed:
  - step-0 CE  (~8.5 = ln(5000), confirms fresh init)
  - step-300 param-norm movement (confirms fp32 master is actually updating)
  - CE / depth_loss / recall_loss every 1000 steps
Saves best.pt (by EMA-CE) and last.pt to checkpoints/binding_run/.
"""

import os
import sys
os.environ["WANDB_MODE"] = "disabled"

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import (
    Config, ModelConfig, MemoryConfig, LorenzConfig, RoutingConfig, TrainConfig,
)
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

torch.manual_seed(42)

SEQ_LEN  = 256
STEPS    = 2500          # batch=64 sees ~160k seqs (= orig batch16x8000 plan), fits ~3h budget
W_DEPTH  = 0.005
W_RECALL = 0.02
P_SYNTH  = 0.5          # fraction of batches drawn from the synthetic binding set
CKPT_DIR = "checkpoints/binding_run"

SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"


def binding_config(batch_size: int) -> Config:
    return Config(
        model=ModelConfig(
            dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
            vocab_size=5000, max_seq_len=SEQ_LEN,
        ),
        memory=MemoryConfig(slots=16, alpha=0.9, beta=0.1),
        lorenz=LorenzConfig(),
        routing=RoutingConfig(temp_start=2.0, temp_end=0.5,
                              anneal_steps=STEPS, max_hops=6),
        train=TrainConfig(
            batch_size=batch_size, lr=2e-4, lr_min=1e-5, weight_decay=1e-3,
            grad_clip=1.0, warmup_steps=200, steps=STEPS,
            use_bf16=True, use_compile=False,
            entropy_weight=0.05, balance_weight=0.001,
        ),
    )


# ── mixed dataset: 50/50 synthetic binding + tinystories ──────────────────────
class MixedBindingDataset(Dataset):
    def __init__(self, sp, seq_len, p_synth=0.5):
        self.seq_len  = seq_len
        self.p_synth  = p_synth

        print(f"  tokenizing {SYNTH_PATH} ...", flush=True)
        with open(SYNTH_PATH, "r", encoding="utf-8") as f:
            synth = sp.EncodeAsIds(f.read())
        print(f"  tokenizing {TS_PATH} ...", flush=True)
        with open(TS_PATH, "r", encoding="utf-8") as f:
            ts = sp.EncodeAsIds(f.read())

        self.synth = torch.tensor(synth, dtype=torch.long)
        self.ts    = torch.tensor(ts,    dtype=torch.long)
        self.n_synth = len(self.synth) // (seq_len + 1)
        self.n_ts    = len(self.ts)    // (seq_len + 1)
        print(f"  synth tokens={len(self.synth):,} blocks={self.n_synth:,}", flush=True)
        print(f"  ts    tokens={len(self.ts):,} blocks={self.n_ts:,}", flush=True)

    def __len__(self):
        return self.n_synth + self.n_ts

    def _block(self, src, n_blocks):
        b = torch.randint(0, n_blocks, (1,)).item()
        start = b * (self.seq_len + 1)
        chunk = src[start: start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    def __getitem__(self, idx):
        if torch.rand(1).item() < self.p_synth:
            return self._block(self.synth, self.n_synth)
        return self._block(self.ts, self.n_ts)


def build_loader(sp, cfg):
    ds = MixedBindingDataset(sp, cfg.model.max_seq_len, P_SYNTH)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)


def run(batch_size):
    cfg    = binding_config(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Config : dim={cfg.model.dim}, seq_len={cfg.model.max_seq_len}, "
          f"batch={cfg.train.batch_size}, steps={STEPS}")
    print(f"Weights: w_depth={W_DEPTH}, w_recall={W_RECALL}  | mix p_synth={P_SYNTH}")

    # confirm Ring 0 fix is live (no `cr == 0` term in exit)
    import inspect
    src = inspect.getsource(SpiderWeb.forward)
    ring0_ok = "exit_mask = decisions == 2" in src and "(cr == 0)" not in src.split("Previously")[0]
    print(f"Ring 0 fix live: {ring0_ok}")
    print()

    model = SpiderWeb(cfg).to(device)
    # fp32 master: params stay float32; only autocast does bf16 compute
    assert next(model.parameters()).dtype == torch.float32, "params must be fp32 master"
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Params : {total_p/1e6:.2f}M (fresh init, fp32 master)\n")

    loss_fn   = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95), fused=(device.type == "cuda"),
    )
    lr_sched  = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)

    on_cuda      = device.type == "cuda"
    autocast_ctx = (torch.autocast("cuda", torch.bfloat16)
                    if on_cuda and cfg.train.use_bf16 else nullcontext())

    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    loader    = build_loader(sp, cfg)
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ── param norm snapshot (for step-300 movement check) ────────────────────
    def param_norm():
        with torch.no_grad():
            return sum(p.float().norm().item() ** 2 for p in model.parameters()) ** 0.5
    norm0 = param_norm()

    os.makedirs(CKPT_DIR, exist_ok=True)
    model.train()
    ema_loss  = None
    best_ema  = float("inf")
    step0_ce  = None

    print(f"{'─'*64}")
    print(f"{'Step':>6}  {'CE':>8}  {'EMA-CE':>8}  {'Depth':>9}  {'Recall':>8}  {'tau':>5}")
    print(f"{'─'*64}")

    for step in range(STEPS):
        x, y = next_batch()
        x, y = x.to(device), y.to(device)
        tau  = tau_sched.get_temp(step)

        with autocast_ctx:
            out = model(x, tau=tau, hard=False)
            if cfg.train.use_bf16 and on_cuda:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=cfg.train.entropy_weight,
                                 w_depth=W_DEPTH, w_recall=W_RECALL)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⛔  NaN/Inf at step {step}. Stopping.")
            return True

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

        if step == 0:
            step0_ce = ce
            print(f"{step:>6}  {ce:>8.4f}  {ema_loss:>8.4f}  "
                  f"{mets['depth']:>9.4f}  {mets['recall']:>8.4f}  {tau:>5.2f}", flush=True)
            print(f"   ↳ step-0 CE check: {ce:.3f}  "
                  f"({'OK ~8.5 fresh' if 7.5 < ce < 9.5 else 'UNEXPECTED'})", flush=True)

        if step == 300:
            norm300 = param_norm()
            print(f"   ↳ step-300 param-norm: {norm0:.3f} -> {norm300:.3f} "
                  f"(Δ={norm300-norm0:+.3f})  "
                  f"{'weights moving (fp32 master live)' if abs(norm300-norm0) > 1e-3 else 'NO MOVEMENT'}",
                  flush=True)

        if step > 0 and (step % 500 == 0 or step == STEPS - 1):
            print(f"{step:>6}  {ce:>8.4f}  {ema_loss:>8.4f}  "
                  f"{mets['depth']:>9.4f}  {mets['recall']:>8.4f}  {tau:>5.2f}", flush=True)

        # save best by EMA-CE (after warmup), and a rolling last.pt
        if step > cfg.train.warmup_steps and ema_loss < best_ema:
            best_ema = ema_loss
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "step": step + 1, "ema_ce": ema_loss,
                        "w_depth": W_DEPTH, "w_recall": W_RECALL},
                       f"{CKPT_DIR}/best.pt")

        if step > 0 and step % 500 == 0:
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "step": step + 1, "ema_ce": ema_loss},
                       f"{CKPT_DIR}/last.pt")

    # final last.pt
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "step": STEPS, "ema_ce": ema_loss}, f"{CKPT_DIR}/last.pt")

    print(f"\n✓  Done. step0_CE={step0_ce:.3f}  final_EMA_CE={ema_loss:.3f}  "
          f"best_EMA_CE={best_ema:.3f}")
    print(f"✓  Saved {CKPT_DIR}/best.pt and {CKPT_DIR}/last.pt")

    nan_params = [n for n, p in model.named_parameters() if p.isnan().any()]
    print("✓  No NaN in params" if not nan_params else f"⛔ NaN in: {nan_params}")
    return False


def main():
    for bs in (64, 32, 16):
        try:
            run(bs)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n⚠️  CUDA OOM at batch={bs}. Falling back to batch={bs//2}.\n", flush=True)
    print("⛔  OOM even at batch=4.")


if __name__ == "__main__":
    main()
