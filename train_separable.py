"""
train_separable.py — write-side redesign A/B (separable entity memory vs blend).

Warm-starts from the competent CE-4.4 baseline (checkpoints/final_run/best.pt)
and fine-tunes on the binding + TinyStories mix. The ONLY thing that changes
between the two arms is one flag:

    --write_mode separable   -> the new causal, content-routed, gated, NON-
                                blending entity memory (SeparableMemoryRead) is
                                active after the hop loop.
    --write_mode blend       -> that module is OFF; behaviour == baseline.

Checkpoints (model + optimizer + scheduler + step) are written every
--save_every steps AND as last.pt, so a run can be stopped and resumed cleanly
with --resume. CE + gate/routing stats are logged to a CSV every 50 steps.

Examples
--------
  # real runs (launch yourself):
  python3 train_separable.py --write_mode separable --steps 3000
  python3 train_separable.py --write_mode blend     --steps 3000

  # resume after a stop:
  python3 train_separable.py --write_mode separable --steps 3000 --resume

  # 2-step dry run (verify only; writes to a throwaway dir):
  python3 train_separable.py --write_mode separable --steps 2 --save_every 1 \
          --out /tmp/sep_dry
"""

import os
import sys
import csv
import argparse
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

SEQ_LEN    = 256
P_SYNTH    = 0.5
BASE_CKPT  = "checkpoints/final_run/best.pt"
SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"

# new modules absent from the baseline checkpoint: allowed to start fresh
NEW_PREFIXES = ("separable_mem.", "query_read.", "struct_read.", "recall_proj.")


def ft_config(batch_size, slots, write_mode, steps) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN,
                          use_struct_read=False, use_query_read=False),
        memory=MemoryConfig(slots=slots, alpha=0.9, beta=0.1, write_mode=write_mode),
        lorenz=LorenzConfig(),
        routing=RoutingConfig(temp_start=0.3, temp_end=0.1,
                              anneal_steps=steps, max_hops=6),
        train=TrainConfig(batch_size=batch_size, lr=1e-4, lr_min=1e-5,
                          weight_decay=1e-3, grad_clip=1.0, warmup_steps=50,
                          steps=steps, use_bf16=True, use_compile=False,
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


def run(args, batch_size):
    write_mode = args.write_mode
    out_dir = os.path.join(args.out, write_mode)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume = args.resume and os.path.exists(os.path.join(out_dir, "last.pt"))
    ckpt_path = os.path.join(out_dir, "last.pt") if resume else BASE_CKPT
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    start_step = int(ckpt.get("step", 0)) if resume else 0

    cfg   = ft_config(batch_size, slots, write_mode, args.steps)
    model = SpiderWeb(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(NEW_PREFIXES)]
    assert not bad, f"unexpected missing (not a new module): {bad}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert next(model.parameters()).dtype == torch.float32

    n_params  = sum(p.numel() for p in model.parameters())
    n_sep     = sum(p.numel() for p in model.separable_mem.parameters())
    n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            resume_kind = "exact (optimizer+scheduler restored)"
        else:
            for _ in range(start_step):
                lr_sched.step()
            resume_kind = "warm (weights only; fresh optimizer)"
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

    # ── BANNER ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  TRAIN_SEPARABLE  write_mode={write_mode}  "
          f"(separable_mem {'ACTIVE' if write_mode=='separable' else 'OFF (baseline)'})",
          flush=True)
    print(f"  device={device}  batch={batch_size}  slots={slots}  seq_len={SEQ_LEN}  "
          f"steps={args.steps}", flush=True)
    print(f"  start={'RESUME '+resume_kind+f' @step {start_step}' if resume else 'WARM from '+BASE_CKPT}",
          flush=True)
    print(f"  params: total={n_params:,}  trainable={n_train:,}  "
          f"separable_mem={n_sep:,}", flush=True)
    print(f"  csv -> {csv_path}   ckpts -> {out_dir}/", flush=True)
    print(f"{'='*70}", flush=True)

    new_csv = not (resume and os.path.exists(csv_path))
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if new_csv:
        csv_w.writerow(["step", "ce", "ema_ce", "tau", "lr",
                        "gate_mean", "gate_frac_on", "route_entropy", "fuse_mean"])
        csv_f.flush()

    def save_ckpt(path, step, ema_val):
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_sched.state_dict(),
                    "step": step, "ema_ce": ema_val,
                    "write_mode": write_mode}, path)

    model.train()
    ema = float(ckpt["ema_ce"]) if resume and "ema_ce" in ckpt else None
    best = ema if ema is not None else float("inf")
    step0_ce = None

    print(f"[{write_mode}] {'Step':>6} {'CE':>8} {'EMA':>8} {'tau':>5}  "
          f"{'gate':>6} {'on%':>5} {'Hroute':>6}", flush=True)
    for step in range(start_step, args.steps):
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
            print(f"[{write_mode}] NaN/Inf at step {step}. Stopping."); csv_f.close(); return True

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step(); lr_sched.step()

        ce = mets["ce"]
        ema = ce if ema is None else 0.95 * ema + 0.05 * ce
        s = out["sep_stats"] or {}
        gm, gf = s.get("gate_mean", ""), s.get("gate_frac_on", "")
        he, fm = s.get("route_entropy", ""), s.get("fuse_mean", "")
        lr_now = lr_sched.get_last_lr()[0]

        if step == start_step:
            step0_ce = ce
            # warm fine-tune CE sits ~5.1 here (tau=0.3 stochastic routing +
            # binding-heavy mix); a fresh init would be ~8.5. <6.5 == warm.
            tag = "WARM ✓ (loaded baseline; fresh would be ~8.5)" if ce < 6.5 \
                  else "NOT WARM (fresh ~8.5)!"
            print(f"[{write_mode}] {step:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f}  "
                  f"{('%.3f'%gm) if gm!='' else '  -  ':>6} "
                  f"{('%.2f'%gf) if gf!='' else ' - ':>5} "
                  f"{('%.3f'%he) if he!='' else '  -  ':>6}", flush=True)
            print(f"[{write_mode}]   step-{start_step} CE = {ce:.3f}  {tag}", flush=True)

        if step % 50 == 0 or step == args.steps - 1:
            csv_w.writerow([step, f"{ce:.5f}", f"{ema:.5f}", f"{tau:.4f}", f"{lr_now:.3e}",
                            (f"{gm:.5f}" if gm != "" else ""),
                            (f"{gf:.5f}" if gf != "" else ""),
                            (f"{he:.5f}" if he != "" else ""),
                            (f"{fm:.5f}" if fm != "" else "")])
            csv_f.flush()
        if step > start_step and step % 100 == 0:
            print(f"[{write_mode}] {step:>6} {ce:>8.4f} {ema:>8.4f} {tau:>5.2f}  "
                  f"{('%.3f'%gm) if gm!='' else '  -  ':>6} "
                  f"{('%.2f'%gf) if gf!='' else ' - ':>5} "
                  f"{('%.3f'%he) if he!='' else '  -  ':>6}", flush=True)

        if step > start_step and (step % args.save_every == 0 or step == args.steps - 1):
            save_ckpt(os.path.join(out_dir, "last.pt"), step + 1, ema)
        if step > cfg.train.warmup_steps and ema < best:
            best = ema
            save_ckpt(os.path.join(out_dir, "best.pt"), step + 1, ema)

    save_ckpt(os.path.join(out_dir, "last.pt"), args.steps, ema)
    csv_f.close()
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[{write_mode}] DONE step0_CE={step0_ce:.3f} final_EMA={ema:.3f} "
          f"best_EMA={best:.3f} -> {out_dir}  {'no NaN' if not nanp else 'NaN:'+str(nanp)}",
          flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_mode", choices=["separable", "blend"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto (48->32->16)")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--out", default="checkpoints/separable_run")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    batches = [args.batch] if args.batch else [48, 32, 16]
    for bs in batches:
        try:
            run(args, bs); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{args.write_mode}] OOM at batch={bs}, falling back.", flush=True)
    print(f"[{args.write_mode}] OOM even at smallest batch.")


if __name__ == "__main__":
    main()
