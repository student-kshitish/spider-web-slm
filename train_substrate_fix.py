"""
Step 2 (substrate fix-and-measure) — short retrain so the 3 fixes take effect.

Warm-start from the hybrid_full checkpoint with ALL three substrate fixes ON:
  undetach_mem=True, residual_stream=True, sharp_head=True.
~1500 steps, same binding+TinyStories 50/50 mix, seq_len=256. Saves to
checkpoints/substrate_fix/. Watch for NaN / loss blow-up — un-detaching writes
and dropping spectral norm both risk it; if it destabilises that is itself the
result (the fixes fight the chaos routing).

Warm-start detail: sharp_head removes spectral_norm, which changes the module
parametrization (checkpoint stores weight_orig, not weight). We BAKE the trained
effective (spectral-normalised) weights into the plain Linears, so the FFN/head
start exactly where hybrid_full left them rather than re-initialising.

Usage:  python3 train_substrate_fix.py [--resume]
"""
import os, sys
os.environ["WANDB_MODE"] = "disabled"
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized
import sentencepiece as spm

from config import (Config, ModelConfig, MemoryConfig, LorenzConfig,
                    RoutingConfig, TrainConfig)
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

SEQ_LEN = 256
STEPS   = 1500
P_SYNTH = 0.5
BASE_CKPT  = "checkpoints/hybrid_full/best.pt"
SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"
OUT        = "checkpoints/substrate_fix"
NEW_PREFIXES = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj")


def ft_config(batch_size, slots, sharp_head, residual_stream, undetach_mem) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN,
                          use_hybrid=True, lookback_width=-1,
                          sharp_head=sharp_head, residual_stream=residual_stream,
                          undetach_mem=undetach_mem),
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


def warm_transfer(B, ckpt_state, slots, device):
    """Load hybrid_full into an ORIGINAL-arch temp, bake the spectral read/head
    weights to plain effective weights, then transfer into the sharp model B."""
    tcfg = ft_config(B.cfg.train.batch_size, slots, sharp_head=False,
                     residual_stream=False, undetach_mem=False)
    temp = SpiderWeb(tcfg).to(device)
    miss, unexp = temp.load_state_dict(ckpt_state, strict=False)
    bad = [k for k in miss if not k.startswith(NEW_PREFIXES)]
    assert not bad and not unexp, f"temp load: missing={bad} unexpected={unexp}"
    # bake spectral_norm on the read/head path -> plain effective weights
    for ring in temp.rings:
        for node in ring:
            for attr in ("w1", "w2", "w3", "aux_proj"):
                mod = getattr(node, attr)
                if is_parametrized(mod, "weight"):
                    remove_parametrizations(mod, "weight", leave_parametrized=True)
    if hasattr(temp.lm_head, "weight_orig"):       # old-API spectral_norm
        torch.nn.utils.remove_spectral_norm(temp.lm_head)
    # now temp keys match B (plain read/head, parametrized router/memory)
    miss2, unexp2 = B.load_state_dict(temp.state_dict(), strict=False)
    bad2 = [k for k in miss2 if not k.startswith(NEW_PREFIXES)]
    assert not bad2 and not unexp2, f"B load: missing={bad2} unexpected={unexp2}"
    del temp


def run(batch_size, resume=False):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{OUT}/last.pt" if resume else BASE_CKPT
    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg = ft_config(batch_size, slots, sharp_head=True,
                    residual_stream=True, undetach_mem=True)
    start_step = int(ckpt.get("step", 0)) if resume else 0

    model = SpiderWeb(cfg).to(device)
    if resume:
        m, u = model.load_state_dict(state, strict=False)
        bad = [k for k in m if not k.startswith(NEW_PREFIXES)]
        assert not bad and not u, f"resume load: missing={bad} unexpected={u}"
    else:
        warm_transfer(model, state, slots, device)
    assert next(model.parameters()).dtype == torch.float32
    print(f"[sfix] device={device} batch={batch_size} slots={slots} "
          f"FIXES: undetach+residual+sharp  "
          f"{'RESUME '+ckpt_path+f' step {start_step}' if resume else 'WARM(baked) from '+BASE_CKPT}",
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
                    "fixes": "undetach+residual+sharp"}, path)

    print(f"[sfix] {'Step':>6} {'CE':>8} {'EMA':>8} {'tau':>5} | {'fire%':>6} "
          f"{'pmax':>5} {'gnorm':>7}", flush=True)
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
            print(f"[sfix] *** NaN/Inf at step {step}. DESTABILISED — "
                  f"the fixes fight the chaos routing. Stopping. ***", flush=True)
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
        hs = out["hybrid_stats"]
        fire = 100 * hs["gate_frac_on"] if hs else float("nan")
        with torch.no_grad():
            pmax = float(out["logits"][:, -1, :].float().softmax(-1).max(-1).values.mean())

        def _line(s):
            print(f"[sfix] {s:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f} | "
                  f"{fire:>5.1f}% {pmax:>5.3f} {float(gnorm):>7.2f}", flush=True)

        if step == start_step:
            step0 = ce; _line(step)
            print(f"[sfix]   step-{start_step} CE = {ce:.3f}  "
                  f"({'expected jump — residual changes activations' if not resume else 'resume'})",
                  flush=True)
        if step > start_step and (step % 250 == 0 or step == STEPS - 1):
            _line(step); save(f"{OUT}/last.pt", step + 1, ema)
        if step > cfg.train.warmup_steps and ema < best:
            best = ema; save(f"{OUT}/best.pt", step + 1, ema)

    save(f"{OUT}/last.pt", STEPS, ema)
    print(f"[sfix] DONE step0_CE={step0:.3f} final_EMA={ema:.3f} best_EMA={best:.3f} "
          f"-> {OUT}", flush=True)
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[sfix] {'no NaN in weights' if not nanp else 'NaN weights: '+str(nanp)}", flush=True)
    return False


def main():
    resume = "--resume" in sys.argv[1:]
    for bs in (32, 24, 16):
        try:
            run(bs, resume=resume); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[sfix] OOM at batch={bs}, falling back.", flush=True)
    print("[sfix] OOM even at batch=16.")


if __name__ == "__main__":
    main()
