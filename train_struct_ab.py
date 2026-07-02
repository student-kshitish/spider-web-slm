"""
Step 2 — Direction-gated structural retrieval A/B fine-tune.

Both arms start from checkpoints/final_run/best.pt and use the position-resolved
structural read (no time-collapse). They differ ONLY by the structural mask:

  Arm A (struct mask ON)  : use_struct_read=True, struct_mask=True  -> checkpoints/struct_on
  Arm B (flat, mask OFF)  : use_struct_read=True, struct_mask=False -> checkpoints/struct_off

Same 50/50 binding+TinyStories mix, seq_len=256, lr=1e-4, 2500 steps, same seed,
CE-only loss (no fixed-lag recall).

Usage:  python3 train_struct_ab.py on
        python3 train_struct_ab.py off

        # Warm-restart from a previous run's checkpoints/<arm>/last.pt and
        # continue to STEPS (fresh optimizer if the ckpt has weights only):
        python3 train_struct_ab.py on  --resume
        python3 train_struct_ab.py off --resume

Checkpoints now persist optimizer + scheduler + step, and last.pt is written
every 250 steps, so an interrupted run can be resumed exactly next time.
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

ARMS = {
    "on":  dict(struct_mask=True,  out="checkpoints/struct_on"),
    "off": dict(struct_mask=False, out="checkpoints/struct_off"),
}

ALLOWED_MISSING = {
    "recall_proj.weight",
    "query_read.ring_bias", "query_read.q_proj.weight", "query_read.o_proj.weight",
    "query_read.gate_proj.weight", "query_read.gate_proj.bias",
    "struct_read.ring_bias", "struct_read.q_proj.weight", "struct_read.o_proj.weight",
    "struct_read.gate_proj.weight", "struct_read.gate_proj.bias",
}


def ft_config(batch_size, slots, struct_mask) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN,
                          use_struct_read=True, struct_mask=struct_mask),
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
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{spec['out']}/last.pt" if resume else BASE_CKPT
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg   = ft_config(batch_size, slots, spec["struct_mask"])

    start_step = int(ckpt.get("step", 0)) if resume else 0

    model = SpiderWeb(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) <= ALLOWED_MISSING, f"unexpected missing: {set(missing)-ALLOWED_MISSING}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert next(model.parameters()).dtype == torch.float32

    print(f"[{arm}] device={device} batch={batch_size} slots={slots} "
          f"use_struct_read=True struct_mask={spec['struct_mask']} "
          f"{'RESUME from '+ckpt_path+f' (day-1 step {start_step})' if resume else 'FRESH from '+BASE_CKPT}",
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
            print(f"[{arm}] restored optimizer+scheduler state (exact resume)", flush=True)
        else:
            for _ in range(start_step):
                lr_sched.step()
            print(f"[{arm}] weights-only ckpt: fresh optimizer, LR fast-forwarded to "
                  f"step {start_step} (warm restart)", flush=True)
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
    seed_ema = float(ckpt["ema_ce"]) if resume and "ema_ce" in ckpt else None
    ema = seed_ema
    best = seed_ema if seed_ema is not None else float("inf")
    step0_ce = None

    def save_ckpt(path, step, ema_val, bias):
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_sched.state_dict(),
                    "step": step, "ema_ce": ema_val, "arm": arm,
                    "struct_mask": spec["struct_mask"], "struct_read_bias": bias}, path)

    print(f"[{arm}] {'Step':>6} {'CE':>8} {'EMA':>8} {'tau':>5} | lookback | s_bias", flush=True)
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
        sb = [round(v, 3) for v in out["struct_read_bias"].tolist()]
        lb = out["struct_lookback"]
        if step == start_step:
            step0_ce = ce
            if resume:
                if ce < 4.8:
                    warm = "RESUME OK ✓ (~4.2 — continuing from day-1 weights)"
                elif ce < 6.5:
                    warm = "WRONG: ~5.3 = reloaded ORIGINAL baseline, not day-1!"
                else:
                    warm = "WRONG: fresh init (~8.5)!"
            else:
                warm = "WARM ✓ (~5)" if ce < 6.5 else "NOT WARM (>6.5)!"
            print(f"[{arm}] {step:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f} | {lb:.3f}    | {sb}", flush=True)
            print(f"[{arm}]   step-{start_step} CE = {ce:.3f}  {warm}", flush=True)
        if step > start_step and (step % 500 == 0 or step == STEPS - 1):
            print(f"[{arm}] {step:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f} | {lb:.3f}    | {sb}", flush=True)
        if step > start_step and (step % 250 == 0 or step == STEPS - 1):
            save_ckpt(f"{spec['out']}/last.pt", step + 1, ema,
                      out["struct_read_bias"].cpu().tolist())
        if step > cfg.train.warmup_steps and ema < best:
            best = ema
            save_ckpt(f"{spec['out']}/best.pt", step + 1, ema,
                      out["struct_read_bias"].cpu().tolist())

    final_bias = out["struct_read_bias"].cpu().tolist()
    save_ckpt(f"{spec['out']}/last.pt", STEPS, ema, final_bias)
    print(f"[{arm}] DONE step0_CE={step0_ce:.3f} final_EMA={ema:.3f} best_EMA={best:.3f} "
          f"lookback={out['struct_lookback']:.3f} s_bias={[round(v,3) for v in final_bias]} "
          f"-> {spec['out']}", flush=True)
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[{arm}] {'no NaN' if not nanp else 'NaN: '+str(nanp)}", flush=True)
    return False


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv[1:]
    arm = pos[0] if pos else "on"
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
