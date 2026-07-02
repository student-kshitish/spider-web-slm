"""
Step 3 — corrected decisive test: fine-tune-from-baseline A/B.

Both arms START from checkpoints/final_run/best.pt (the CE 4.42 model that
already knows English), so any binding gain is attributable to the mechanism,
not to relearning language.

  Common : 50/50 binding+TinyStories mix, seq_len=256, lr=1e-4, ~2000 steps,
           identical seed/data order.
  Arm A (on)  : w_depth=0.005, w_recall=0.02 -> checkpoints/binding_ft_on/
  Arm B (off) : w_depth=0.0,   w_recall=0.0  -> checkpoints/binding_ft_off/

Usage:  python3 train_binding_ft.py on
        python3 train_binding_ft.py off

Confirms step-0 CE ~4.4 (warm start, NOT 8.5).
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

SEQ_LEN  = 256
STEPS    = 2000
P_SYNTH  = 0.5
BASE_CKPT = "checkpoints/final_run/best.pt"
SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"

ARMS = {
    "on":  dict(w_depth=0.005, w_recall=0.02, out="checkpoints/binding_ft_on"),
    "off": dict(w_depth=0.0,   w_recall=0.0,  out="checkpoints/binding_ft_off"),
}


def ft_config(batch_size, slots) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN),
        memory=MemoryConfig(slots=slots, alpha=0.9, beta=0.1),
        lorenz=LorenzConfig(),
        # low, gentle temperature: baseline was trained to tau~0.1, keep routing stable
        routing=RoutingConfig(temp_start=0.3, temp_end=0.1,
                              anneal_steps=STEPS, max_hops=6),
        train=TrainConfig(batch_size=batch_size, lr=1e-4, lr_min=1e-5,
                          weight_decay=1e-3, grad_clip=1.0, warmup_steps=50,
                          steps=STEPS, use_bf16=True, use_compile=False,
                          entropy_weight=0.05, balance_weight=0.001),
    )


class MixedBindingDataset(Dataset):
    def __init__(self, sp, seq_len, p_synth=0.5):
        self.seq_len = seq_len
        self.p_synth = p_synth
        with open(SYNTH_PATH, "r", encoding="utf-8") as f:
            synth = sp.EncodeAsIds(f.read())
        with open(TS_PATH, "r", encoding="utf-8") as f:
            ts = sp.EncodeAsIds(f.read())
        self.synth = torch.tensor(synth, dtype=torch.long)
        self.ts    = torch.tensor(ts,    dtype=torch.long)
        self.n_synth = len(self.synth) // (seq_len + 1)
        self.n_ts    = len(self.ts)    // (seq_len + 1)

    def __len__(self):
        return self.n_synth + self.n_ts

    def _block(self, src, n):
        b = torch.randint(0, n, (1,)).item()
        s = b * (self.seq_len + 1)
        chunk = src[s: s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    def __getitem__(self, idx):
        if torch.rand(1).item() < self.p_synth:
            return self._block(self.synth, self.n_synth)
        return self._block(self.ts, self.n_ts)


def detect_slots(state):
    return state["rings.0.0.memory.m_t_seed"].shape[0]


def run(arm, batch_size):
    spec = ARMS[arm]
    torch.manual_seed(42)                       # identical data order across arms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load competent baseline ───────────────────────────────────────────────
    ckpt  = torch.load(BASE_CKPT, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = detect_slots(state)
    cfg   = ft_config(batch_size, slots)

    model = SpiderWeb(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # recall_proj is new (not in baseline) -> fresh init; only matters for arm "on"
    assert missing == ["recall_proj.weight"] or missing == [], f"unexpected missing: {missing}"
    assert next(model.parameters()).dtype == torch.float32

    print(f"[{arm}] device={device} batch={batch_size} slots={slots} "
          f"w_depth={spec['w_depth']} w_recall={spec['w_recall']}", flush=True)
    print(f"[{arm}] loaded {BASE_CKPT} (warm start); missing={missing}", flush=True)

    loss_fn   = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay,
                                  betas=(0.9, 0.95), fused=(device.type == "cuda"))
    lr_sched  = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)
    on_cuda   = device.type == "cuda"
    ac = (torch.autocast("cuda", torch.bfloat16)
          if on_cuda and cfg.train.use_bf16 else nullcontext())

    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    ds = MixedBindingDataset(sp, cfg.model.max_seq_len, P_SYNTH)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    os.makedirs(spec["out"], exist_ok=True)
    model.train()
    ema = None
    best = float("inf")
    step0_ce = None

    print(f"[{arm}] {'Step':>6} {'CE':>8} {'EMA':>8} {'Depth':>9} {'Recall':>8} {'tau':>5}", flush=True)
    for step in range(STEPS):
        x, y = next_batch()
        x, y = x.to(device), y.to(device)
        tau  = tau_sched.get_temp(step)
        with ac:
            out = model(x, tau=tau, hard=False)
            if on_cuda and cfg.train.use_bf16:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=cfg.train.entropy_weight,
                                 w_depth=spec["w_depth"], w_recall=spec["w_recall"])
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[{arm}] NaN/Inf at step {step}. Stopping."); return True

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step(); lr_sched.step()

        ce = mets["ce"]
        ema = ce if ema is None else 0.95 * ema + 0.05 * ce
        if step == 0:
            step0_ce = ce
            warm = "WARM ✓ (~4.4 baseline)" if ce < 6.0 else "NOT WARM (>6) — check load!"
            print(f"[{arm}] {step:>6} {ce:>8.4f} {ema:>8.4f} {mets['depth']:>9.4f} "
                  f"{mets['recall']:>8.4f} {tau:>5.2f}", flush=True)
            print(f"[{arm}]   step-0 CE = {ce:.3f}  {warm}", flush=True)
        if step > 0 and (step % 250 == 0 or step == STEPS - 1):
            print(f"[{arm}] {step:>6} {ce:>8.4f} {ema:>8.4f} {mets['depth']:>9.4f} "
                  f"{mets['recall']:>8.4f} {tau:>5.2f}", flush=True)
        if step > cfg.train.warmup_steps and ema < best:
            best = ema
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "step": step + 1, "ema_ce": ema, "arm": arm,
                        "w_depth": spec["w_depth"], "w_recall": spec["w_recall"]},
                       f"{spec['out']}/best.pt")

    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "step": STEPS, "ema_ce": ema, "arm": arm}, f"{spec['out']}/last.pt")
    print(f"[{arm}] DONE step0_CE={step0_ce:.3f} final_EMA={ema:.3f} best_EMA={best:.3f} "
          f"-> {spec['out']}", flush=True)
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[{arm}] {'no NaN' if not nanp else 'NaN: '+str(nanp)}", flush=True)
    return False


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "on"
    assert arm in ARMS, f"arm must be one of {list(ARMS)}"
    for bs in (48, 32, 16):   # 48/arm => ~2.4GB each, safe for two parallel arms
        try:
            run(arm, bs); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{arm}] OOM at batch={bs}, falling back.", flush=True)
    print(f"[{arm}] OOM even at batch=16.")


if __name__ == "__main__":
    main()
