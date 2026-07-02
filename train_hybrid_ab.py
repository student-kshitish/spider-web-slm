"""
Step 3 — Unified-hybrid A/B(/C) fine-tune (only after Step 2 passed).

Three arms, all warm-started from checkpoints/final_run/best.pt, same 50/50
binding+TinyStories mix, seq_len=256, lr=1e-4, 2500 steps, same seed, CE-only
loss (no depth/recall aux). They differ ONLY in the hybrid lookback config:

  bounded : use_hybrid=True,  lookback_width=32   -> checkpoints/hybrid_bounded
  full    : use_hybrid=True,  lookback_width=-1   -> checkpoints/hybrid_full   (all prior)
  off     : use_hybrid=False  (pure Spider Web)   -> checkpoints/hybrid_off    (baseline)

The "off" arm is the control: architecturally identical forward minus the gated
lookback module, so any CE / binding gap is attributable to the hybrid attention.

Efficiency note: for the two ON arms we log the gate FIRING RATE (frac of tokens
the gate flags) and the off-self lookback mass — i.e. how often attention was
actually invoked, which is what makes it "surgical" rather than full attention.

Usage:  python3 train_hybrid_ab.py bounded
        python3 train_hybrid_ab.py full
        python3 train_hybrid_ab.py off
        python3 train_hybrid_ab.py bounded --resume   # exact resume from last.pt
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
STEPS    = 2500
P_SYNTH  = 0.5
BASE_CKPT  = "checkpoints/final_run/best.pt"
SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"

# new modules absent from the day-1 checkpoint (loaded strict=False)
NEW_PREFIXES = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj")

ARMS = {
    "bounded": dict(use_hybrid=True,  width=32, out="checkpoints/hybrid_bounded"),
    "full":    dict(use_hybrid=True,  width=-1, out="checkpoints/hybrid_full"),
    "off":     dict(use_hybrid=False, width=32, out="checkpoints/hybrid_off"),
}


def ft_config(batch_size, slots, use_hybrid, width) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN,
                          use_hybrid=use_hybrid, lookback_width=width),
        memory=MemoryConfig(slots=slots, alpha=0.9, beta=0.1),
        lorenz=LorenzConfig(),
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


def run(arm, batch_size, resume=False):
    spec = ARMS[arm]
    use_hybrid, width = spec["use_hybrid"], spec["width"]
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{spec['out']}/last.pt" if resume else BASE_CKPT
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg   = ft_config(batch_size, slots, use_hybrid, width)

    start_step = int(ckpt.get("step", 0)) if resume else 0

    model = SpiderWeb(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(NEW_PREFIXES)]
    assert not bad, f"unexpected missing: {bad}"
    # on a resume, last.pt was saved WITH the new modules, so they are present.
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert next(model.parameters()).dtype == torch.float32

    print(f"[{arm}] device={device} batch={batch_size} slots={slots} "
          f"use_hybrid={use_hybrid} width={'full' if width <= 0 else width} "
          f"{'RESUME from '+ckpt_path+f' (step {start_step})' if resume else 'WARM from '+BASE_CKPT}",
          flush=True)

    loss_fn   = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay,
                                  betas=(0.9, 0.95), fused=(device.type == "cuda"))
    lr_sched  = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)
    if resume:
        if "optimizer" in ckpt and "lr_sched" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_sched.load_state_dict(ckpt["lr_sched"])
            print(f"[{arm}] restored optimizer+scheduler (exact resume)", flush=True)
        else:
            for _ in range(start_step):
                lr_sched.step()
            print(f"[{arm}] weights-only ckpt: fresh optimizer, LR fast-forwarded "
                  f"to step {start_step}", flush=True)
    on_cuda = device.type == "cuda"
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
    seed_ema = float(ckpt["ema_ce"]) if resume and "ema_ce" in ckpt else None
    ema  = seed_ema
    best = seed_ema if seed_ema is not None else float("inf")
    step0_ce = None

    def save_ckpt(path, step, ema_val):
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_sched.state_dict(),
                    "step": step, "ema_ce": ema_val, "arm": arm,
                    "use_hybrid": use_hybrid, "lookback_width": width}, path)

    print(f"[{arm}] {'Step':>6} {'CE':>8} {'EMA':>8} {'tau':>5} | "
          f"{'fire%':>6} {'gate':>5} {'lkbk':>5}", flush=True)
    for step in range(start_step, STEPS):
        x, y = next_batch()
        x, y = x.to(device), y.to(device)
        tau  = tau_sched.get_temp(step)
        with ac:
            out = model(x, tau=tau, hard=False)
            if on_cuda and cfg.train.use_bf16:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=cfg.train.entropy_weight,
                                 w_depth=0.0, w_recall=0.0)
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
        hs = out["hybrid_stats"]
        fire = (100 * hs["gate_frac_on"]) if hs else float("nan")
        gm   = hs["gate_mean"] if hs else float("nan")
        lb   = hs["lookback_frac"] if hs else float("nan")

        def _line(s):
            print(f"[{arm}] {s:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f} | "
                  f"{fire:>5.1f}% {gm:>5.3f} {lb:>5.3f}", flush=True)

        if step == start_step:
            step0_ce = ce
            warm = "WARM ✓ (~5)" if ce < 6.5 else "NOT WARM (>6.5)!"
            _line(step)
            print(f"[{arm}]   step-{start_step} CE = {ce:.3f}  {warm}", flush=True)
        if step > start_step and (step % 500 == 0 or step == STEPS - 1):
            _line(step)
        if step > start_step and (step % 250 == 0 or step == STEPS - 1):
            save_ckpt(f"{spec['out']}/last.pt", step + 1, ema)
        if step > cfg.train.warmup_steps and ema < best:
            best = ema
            save_ckpt(f"{spec['out']}/best.pt", step + 1, ema)

    save_ckpt(f"{spec['out']}/last.pt", STEPS, ema)
    print(f"[{arm}] DONE step0_CE={step0_ce:.3f} final_EMA={ema:.3f} "
          f"best_EMA={best:.3f} fire={fire:.1f}% lookback={lb:.3f} -> {spec['out']}",
          flush=True)
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[{arm}] {'no NaN' if not nanp else 'NaN: '+str(nanp)}", flush=True)
    return False


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv[1:]
    arm = pos[0] if pos else "bounded"
    assert arm in ARMS, f"arm must be one of {list(ARMS)}"
    for bs in (48, 32, 16):
        try:
            run(arm, bs, resume=resume); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{arm}] OOM at batch={bs}, falling back.", flush=True)
    print(f"[{arm}] OOM even at batch=16.")


if __name__ == "__main__":
    main()
