"""
Step 2 (mean-pool fix-and-measure) — short fine-tune with the time mean-pool
KILLED and the separable entity store ENABLED.

Warm-start from checkpoints/substrate_fix/best.pt (which already carries the
three prior fixes: undetach_mem + residual_stream + sharp_head). On top of that:

  no_meanpool = True            # web.py:172 mean(dim=1) removed -> memory is now
                                # POSITION-RESOLVED (B,T,slots,d); an entity
                                # written at position k is NOT averaged across the
                                # whole sequence (the irreversible destroyer the
                                # probe localised).
  write_mode  = "separable"     # SeparableMemoryRead active: one entity per slot,
                                # content-routed, gated, non-blending.

The prior three fixes stay ON. ~1500 steps, binding+TinyStories 50/50, seq_len
256. Saves to checkpoints/nomeanpool_run/.

MEMORY: position-resolved (B,T,slots,d) on the autograd graph (undetach_mem) across
all hops/nodes is ~T x the baseline memory. On an 8 GB GPU this needs a small
batch; we fall back 8 -> 6 -> 4 -> 2.

Removing the mean-pool changes the memory scale, so the first ~100 steps are
watched closely for NaN / blow-up. A NaN is itself a result (the change
destabilises the substrate).

Usage:  python3 train_nomeanpool.py [--resume]
"""
import os, sys
os.environ["WANDB_MODE"] = "disabled"
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import (Config, ModelConfig, MemoryConfig, LorenzConfig,
                    RoutingConfig, TrainConfig)
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

SEQ_LEN = 256
STEPS   = 1500
P_SYNTH = 0.5
BASE_CKPT  = "checkpoints/substrate_fix/best.pt"
SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"
OUT        = "checkpoints/nomeanpool_run"
FIXES      = "undetach+residual+sharp+nomeanpool"
NEW_PREFIXES = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj")


def ft_config(batch_size, slots) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN,
                          use_hybrid=True, lookback_width=-1,
                          sharp_head=True, residual_stream=True,
                          undetach_mem=True, no_meanpool=True),
        memory=MemoryConfig(slots=slots, alpha=0.9, beta=0.1,
                            write_mode="separable"),
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
        self.seq_len, self.p_synth = seq_len, p_synth
        with open(SYNTH_PATH, encoding="utf-8") as f:
            self.synth = torch.tensor(sp.EncodeAsIds(f.read()), dtype=torch.long)
        with open(TS_PATH, encoding="utf-8") as f:
            self.ts = torch.tensor(sp.EncodeAsIds(f.read()), dtype=torch.long)
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


def run(batch_size, resume=False):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{OUT}/last.pt" if resume else BASE_CKPT
    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg = ft_config(batch_size, slots)
    start_step = int(ckpt.get("step", 0)) if resume else 0

    model = SpiderWeb(cfg).to(device)
    # substrate_fix already baked sharp_head (plain lm_head) and contains the
    # separable_mem params (inactive there) -> direct load, no spectral baking.
    m, u = model.load_state_dict(state, strict=False)
    bad = [k for k in m if not k.startswith(NEW_PREFIXES)]
    assert not bad and not u, f"warm load: missing={bad} unexpected={u}"
    assert next(model.parameters()).dtype == torch.float32
    print(f"[nmp] device={device} batch={batch_size} slots={slots} "
          f"FIXES: {FIXES}  write_mode=separable  "
          f"{'RESUME '+ckpt_path+f' step {start_step}' if resume else 'WARM from '+BASE_CKPT}",
          flush=True)

    loss_fn = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay,
                                  betas=(0.9, 0.95), fused=(device.type == "cuda"))
    lr_sched = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)
    if resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"]); lr_sched.load_state_dict(ckpt["lr_sched"])
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
        try:    return next(data_iter)
        except StopIteration:
            data_iter = iter(loader); return next(data_iter)

    os.makedirs(OUT, exist_ok=True)
    model.train()
    ema = float(ckpt["ema_ce"]) if resume and "ema_ce" in ckpt else None
    best = ema if ema is not None else float("inf")
    step0 = None

    def save(path, step, ema_val):
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "lr_sched": lr_sched.state_dict(),
                    "step": step, "ema_ce": ema_val,
                    "fixes": FIXES, "write_mode": "separable",
                    "no_meanpool": True}, path)

    print(f"[nmp] {'Step':>6} {'CE':>8} {'EMA':>8} {'tau':>5} | {'fire%':>6} "
          f"{'pmax':>5} {'gate':>5} {'gnorm':>7}", flush=True)
    for step in range(start_step, STEPS):
        x, y = next_batch(); x, y = x.to(device), y.to(device)
        tau = tau_sched.get_temp(step)
        with ac:
            out = model(x, tau=tau, hard=False)
            if on_cuda and cfg.train.use_bf16:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=cfg.train.entropy_weight,
                                 w_depth=0.0, w_recall=0.0)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[nmp] *** NaN/Inf at step {step}. DESTABILISED — killing the "
                  f"mean-pool changed the memory scale past stability. Stopping. ***",
                  flush=True)
            return True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step(); lr_sched.step()

        ce = mets["ce"]
        ema = ce if ema is None else 0.95 * ema + 0.05 * ce
        hs = out["hybrid_stats"]; ss = out["sep_stats"]
        fire = 100 * hs["gate_frac_on"] if hs else float("nan")
        gate = ss["gate_mean"] if ss else float("nan")
        with torch.no_grad():
            pmax = float(out["logits"][:, -1, :].float().softmax(-1).max(-1).values.mean())

        def _line(s):
            print(f"[nmp] {s:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f} | "
                  f"{fire:>5.1f}% {pmax:>5.3f} {gate:>5.2f} {float(gnorm):>7.2f}",
                  flush=True)

        # watch the first ~100 steps closely (memory-scale change risk window)
        if step == start_step:
            step0 = ce; _line(step)
            print(f"[nmp]   step-{start_step} CE = {ce:.3f}  (expected jump — "
                  f"no_meanpool changes the memory scale)", flush=True)
        elif step <= start_step + 100 and step % 10 == 0:
            _line(step)
        elif step % 250 == 0 or step == STEPS - 1:
            _line(step); save(f"{OUT}/last.pt", step + 1, ema)
        if step > cfg.train.warmup_steps and ema < best:
            best = ema; save(f"{OUT}/best.pt", step + 1, ema)

    save(f"{OUT}/last.pt", STEPS, ema)
    print(f"[nmp] DONE step0_CE={step0:.3f} final_EMA={ema:.3f} best_EMA={best:.3f} "
          f"-> {OUT}", flush=True)
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[nmp] {'no NaN in weights' if not nanp else 'NaN weights: '+str(nanp)}",
          flush=True)
    return False


def main():
    resume = "--resume" in sys.argv[1:]
    for bs in (32, 24, 16, 8):
        try:
            run(bs, resume=resume); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[nmp] OOM at batch={bs}, falling back.", flush=True)
    print("[nmp] OOM even at batch=2.")


if __name__ == "__main__":
    main()
