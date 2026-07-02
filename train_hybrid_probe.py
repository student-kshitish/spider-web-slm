"""
Step 2 — THE DECISIVE CHEAP TEST: is the lookback gate learnable?

Hypothesis under test
---------------------
The separable-write STORAGE gate fired at WRITE time and only paid off much
later at READ time; the signal was DISTANT, the gate got no usable gradient and
collapsed to ~0. The LOOKBACK gate fires at READ time t and pays off
IMMEDIATELY (better prediction of token t+1, same position). So it SHOULD get a
local gradient and learn to fire on tokens that benefit from looking back.

This script does NOT add any auxiliary supervision to the gate. It learns from
the CE signal alone — that is the whole point. We warm-start from
final_run/best.pt (o_proj zero-init -> identity), fine-tune ~800 steps with
bounded N=32 lookback, and watch:

  (1) gate on-rate trajectory      — does it stay meaningfully on / differentiate,
                                      or collapse toward 0 like the storage gate?
  (2) CE on flagged vs unflagged   — are the tokens the gate fires on the ones
                                      that get lower CE? (user's headline metric)
  (3) COUNTERFACTUAL on vs off     — seeded on/off forward on a fixed batch:
                                      does turning attention ON actually LOWER CE
                                      on FLAGGED tokens more than on unflagged?
                                      (the rigorous "attention is helping the
                                      tokens it's invoked on" readout)
  (4) pronoun firing               — does the gate fire MORE on pronoun positions
                                      (he/she/it/they/...) than on average?
  (5) o_proj norm                  — confirms o_proj moved off zero, i.e. the
                                      gate is actually receiving a gradient.

VERDICT at the end:
  gate collapses to ~0 AND no flagged-CE benefit  -> immediate-signal hypothesis
                                                      WRONG. STOP.
  gate fires selectively (pronoun-biased) AND counterfactual helps flagged tokens
                                                   -> hypothesis HOLDS. proceed to
                                                      Step 3 (after user sees this).

Usage:  python3 train_hybrid_probe.py            # bounded N=32, 800 steps
"""

import os
import sys
os.environ["WANDB_MODE"] = "disabled"

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import (
    Config, ModelConfig, MemoryConfig, LorenzConfig, RoutingConfig, TrainConfig,
)
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

SEQ_LEN  = 256
STEPS    = 800
N_WIDTH  = 32
P_SYNTH  = 0.5
LOG_EVERY     = 50
CFACT_EVERY   = 100      # counterfactual on-vs-off eval cadence
BASE_CKPT  = "checkpoints/final_run/best.pt"
SYNTH_PATH = "data/raw/binding.txt"
TS_PATH    = "data/raw/tinystories.txt"
OUT        = "checkpoints/hybrid_probe"

# new modules absent from the day-1 checkpoint (loaded strict=False)
NEW_PREFIXES = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj")

PRONOUNS = {"he", "she", "it", "they", "him", "her", "his", "their",
            "them", "hers", "its", "himself", "herself", "itself"}


def ft_config(batch_size, slots) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=SEQ_LEN,
                          use_hybrid=True, lookback_width=N_WIDTH),
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


def build_pronoun_mask(sp, vocab_size):
    """bool (V,): True for token ids whose piece is a pronoun (any casing)."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(vocab_size):
        piece = sp.IdToPiece(i).replace("▁", "").strip().lower()
        if piece in PRONOUNS:
            mask[i] = True
    return mask


@torch.no_grad()
def counterfactual(model, x, y, V, seed):
    """
    Seeded on-vs-off forward on the SAME batch (same router noise). Returns, for
    flagged and unflagged token sets (by the ON-run gate), the CE drop from
    turning lookback ON:  delta = CE_off - CE_on   (positive = attention helps).
    """
    torch.manual_seed(seed)
    out_off = model(x, tau=0.1, use_hybrid=False)
    torch.manual_seed(seed)
    out_on  = model(x, tau=0.1, use_hybrid=True, lookback_width=N_WIDTH)

    ce_off = F.cross_entropy(out_off["logits"].float().view(-1, V),
                             y.view(-1), reduction="none").view_as(y)
    ce_on  = F.cross_entropy(out_on["logits"].float().view(-1, V),
                             y.view(-1), reduction="none").view_as(y)
    gate = out_on["hybrid_stats"]["gate"]                 # (B,T)
    flag = gate > 0.5
    d = ce_off - ce_on                                    # +ve => ON helps
    def _m(t, msk):
        return float(t[msk].mean()) if msk.any() else float("nan")
    return {
        "d_flag":   _m(d, flag),
        "d_unflag": _m(d, ~flag),
        "ce_on_flag":   _m(ce_on, flag),
        "ce_on_unflag": _m(ce_on, ~flag),
        "frac_on":  float(flag.float().mean()),
    }


def run(batch_size):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(BASE_CKPT, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg   = ft_config(batch_size, slots)

    model = SpiderWeb(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(NEW_PREFIXES)]
    assert not bad, f"unexpected missing: {bad}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert next(model.parameters()).dtype == torch.float32

    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    pron_mask = build_pronoun_mask(sp, cfg.model.vocab_size).to(device)
    print(f"[probe] device={device} batch={batch_size} slots={slots} "
          f"use_hybrid=True width=N={N_WIDTH} steps={STEPS} "
          f"pronoun_ids={int(pron_mask.sum())} WARM from {BASE_CKPT}", flush=True)

    loss_fn   = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay,
                                  betas=(0.9, 0.95), fused=(device.type == "cuda"))
    lr_sched  = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)
    on_cuda   = device.type == "cuda"
    ac = (torch.autocast("cuda", torch.bfloat16)
          if on_cuda and cfg.train.use_bf16 else nullcontext())

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

    # a FIXED eval batch for the counterfactual (kept constant across the run)
    ex, ey = next_batch()
    ex, ey = ex.to(device), ey.to(device)

    os.makedirs(OUT, exist_ok=True)
    model.train()
    V = cfg.model.vocab_size

    print(f"[probe] {'step':>5} {'CE':>7} | {'gate':>5} {'on%':>5} | "
          f"{'CEflag':>7} {'CEunfl':>7} dCE | {'pron%':>6} {'rest%':>6} | "
          f"{'oNorm':>6} {'lkbk':>5}", flush=True)

    traj = []
    for step in range(STEPS):
        x, y = next_batch()
        x, y = x.to(device), y.to(device)
        tau  = tau_sched.get_temp(step)
        with ac:
            out = model(x, tau=tau, hard=False,
                        use_hybrid=True, lookback_width=N_WIDTH)
            if on_cuda and cfg.train.use_bf16:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=cfg.train.entropy_weight,
                                 w_depth=0.0, w_recall=0.0)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[probe] NaN/Inf at step {step}. Stopping."); return True

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step(); lr_sched.step()

        # ── live diagnostics (no_grad) ────────────────────────────────────────
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            with torch.no_grad():
                hs   = out["hybrid_stats"]
                gate = hs["gate"]                                 # (B,T)
                flag = gate > 0.5
                ce_tok = F.cross_entropy(out["logits"].view(-1, V),
                                         y.view(-1), reduction="none").view_as(y)
                ce_flag   = float(ce_tok[flag].mean())  if flag.any()  else float("nan")
                ce_unflag = float(ce_tok[~flag].mean()) if (~flag).any() else float("nan")
                # pronoun firing: mean gate on pronoun input positions vs rest
                pos_pron = pron_mask[x]                           # (B,T)
                g_pron = float(gate[pos_pron].mean()) if pos_pron.any() else float("nan")
                g_rest = float(gate[~pos_pron].mean()) if (~pos_pron).any() else float("nan")

            cf = ""
            cfd = None
            if step % CFACT_EVERY == 0 or step == STEPS - 1:
                model.eval()
                cfd = counterfactual(model, ex, ey, V, seed=1234)
                model.train()
                cf = (f" | cf dFlag={cfd['d_flag']:+.4f} dUnfl={cfd['d_unflag']:+.4f}")

            print(f"[probe] {step:>5} {mets['ce']:>7.4f} | "
                  f"{hs['gate_mean']:>5.3f} {100*hs['gate_frac_on']:>4.1f}% | "
                  f"{ce_flag:>7.4f} {ce_unflag:>7.4f} {ce_unflag-ce_flag:>+.3f} | "
                  f"{100*g_pron:>5.1f}% {100*g_rest:>5.1f}% | "
                  f"{hs['o_proj_norm']:>6.3f} {hs['lookback_frac']:>5.3f}{cf}",
                  flush=True)
            traj.append(dict(step=step, ce=mets["ce"], gate_mean=hs["gate_mean"],
                             on=hs["gate_frac_on"], ce_flag=ce_flag,
                             ce_unflag=ce_unflag, g_pron=g_pron, g_rest=g_rest,
                             o_norm=hs["o_proj_norm"], cf=cfd))

    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "step": STEPS, "traj": traj}, f"{OUT}/last.pt")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    first, last = traj[0], traj[-1]
    cf_last = last["cf"] or {}
    print("\n" + "=" * 72, flush=True)
    print("STEP 2 VERDICT — is the lookback gate learnable?", flush=True)
    print("=" * 72, flush=True)
    print(f"  on-rate     : {100*first['on']:.1f}%  ->  {100*last['on']:.1f}%", flush=True)
    print(f"  gate mean   : {first['gate_mean']:.3f}  ->  {last['gate_mean']:.3f}", flush=True)
    print(f"  o_proj norm : {first['o_norm']:.3f}  ->  {last['o_norm']:.3f}  "
          f"(off-zero => gate received gradient)", flush=True)
    print(f"  pronoun gate: {100*last['g_pron']:.1f}%  vs rest {100*last['g_rest']:.1f}%  "
          f"(selectivity = {last['g_pron']-last['g_rest']:+.3f})", flush=True)
    print(f"  CE flagged  : {last['ce_flag']:.3f}  vs unflagged {last['ce_unflag']:.3f}", flush=True)
    if cf_last:
        print(f"  counterfact : ON lowers CE by {cf_last['d_flag']:+.4f} on FLAGGED "
              f"vs {cf_last['d_unflag']:+.4f} on unflagged tokens", flush=True)

    collapsed = last["on"] < 0.05 and last["o_norm"] < 1e-3
    pron_sel  = (last["g_pron"] - last["g_rest"]) > 0.02
    helps_flag = bool(cf_last) and (cf_last["d_flag"] > cf_last["d_unflag"]) \
                 and (cf_last["d_flag"] > 0)
    print("-" * 72, flush=True)
    if collapsed and not helps_flag:
        print("  >> COLLAPSED like the storage gate. Immediate-signal hypothesis "
              "looks WRONG. STOP.", flush=True)
    elif helps_flag and not collapsed:
        print("  >> Gate fires meaningfully AND attention helps the tokens it is "
              "invoked on.", flush=True)
        print("  >> Immediate-signal hypothesis HOLDS. Cleared to proceed to "
              "Step 3 (pending user sign-off).", flush=True)
        if pron_sel:
            print("  >> Bonus: gate is pronoun-selective (fires more on anaphora).", flush=True)
    else:
        print("  >> MIXED/INCONCLUSIVE. Report numbers, do not auto-proceed.", flush=True)
    print("=" * 72, flush=True)
    return False


def main():
    for bs in (48, 32, 16):
        try:
            run(bs); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[probe] OOM at batch={bs}, falling back.", flush=True)
    print("[probe] OOM even at batch=16.")


if __name__ == "__main__":
    main()
