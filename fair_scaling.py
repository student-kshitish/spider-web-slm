# fair_scaling.py
# Test 1 — Fair parameter-matched memory comparison (Spider Web vs same-capacity transformer)
# Test 2 — Throughput (tokens/second) at each sequence length for both models
#
# Matched transformer: dim=128, heads=4, ff_mult=4, 11 layers, no weight tying
# → ~3.45M params  (Spider Web has 3.48M params, <1% difference)

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb
from infer import INFER_TAU

CKPT     = "checkpoints/final_run/best.pt"
VOCAB    = 5000
SEQ_LENS = [64, 128, 256, 512, 1024, 2048]
N_WARMUP = 3
N_RUNS   = 8


# ══════════════════════════════════════════════════════════════════
# PARAMETER-MATCHED TRANSFORMER
# ══════════════════════════════════════════════════════════════════

class _Block(nn.Module):
    """Pre-norm transformer block: LayerNorm → MHA → residual → LN → FFN → residual."""
    def __init__(self, d: int, heads: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        # bias=False keeps param count predictable; no dropout
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True,
                                            dropout=0.0, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(d * ff_mult, d, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        h, _ = self.attn(xn, xn, xn, attn_mask=mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class MatchedTransformer(nn.Module):
    """
    Standard causal decoder-only transformer.
    d=128, heads=4, ff_mult=4, 11 layers, separate embed + lm_head → 3.45M params.
    Uses explicit upper-triangular causal mask (non-flash), so O(T²) memory is
    faithfully reproduced.
    """
    def __init__(self, vocab: int = VOCAB, d: int = 128,
                 heads: int = 4, layers: int = 11, ff_mult: int = 4):
        super().__init__()
        self.d      = d
        self.embed  = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([_Block(d, heads, ff_mult) for _ in range(layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        T = x.size(1)
        # explicit float causal mask: -inf above diagonal, 0 on and below
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.norm(h))


# ══════════════════════════════════════════════════════════════════
# MEASUREMENT UTILS
# ══════════════════════════════════════════════════════════════════

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure(model_fn, seq_len: int, device: torch.device,
            ctx, n_warmup: int, n_runs: int):
    """
    Returns (activation_mb, throughput_tok_per_sec).

    activation_mb  — peak GPU allocation during forward minus model-param footprint.
    throughput     — seq_len tokens processed per second (single forward, batch=1).
    """
    x = torch.randint(0, VOCAB, (1, seq_len), device=device)

    # warmup: fill caches, JIT, cuBLAS workspaces
    for _ in range(n_warmup):
        with torch.no_grad(), ctx:
            model_fn(x)
    torch.cuda.synchronize(device)

    # snapshot param footprint right after warmup (before reset)
    param_bytes = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad(), ctx:
            model_fn(x)
    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    peak_bytes   = torch.cuda.max_memory_allocated(device)
    activation_mb = max((peak_bytes - param_bytes) / 1024 / 1024, 0.0)

    elapsed_s    = (t_end - t_start) / n_runs
    throughput   = seq_len / elapsed_s   # tokens/sec (single forward pass)

    return activation_mb, throughput


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda  = device.type == "cuda"
    ctx      = torch.autocast("cuda", torch.bfloat16) if on_cuda else nullcontext()

    # ── Spider Web ────────────────────────────────────────────────
    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = max(SEQ_LENS)

    sw = SpiderWeb(cfg).to(device)
    ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in ckpt["model"].items()}
    sw.load_state_dict(state)
    sw.eval()

    def sw_fn(x):
        return sw(x, tau=INFER_TAU, hard=True)["logits"]

    # ── Matched Transformer ───────────────────────────────────────
    mt = MatchedTransformer().to(device)
    mt.eval()

    sw_params = count_params(sw)
    mt_params = count_params(mt)

    print("━" * 72)
    print("Fair Scaling Benchmark — Parameter-matched Spider Web vs Transformer")
    print("━" * 72)
    print(f"  Spider Web        : {sw_params:,} params  ({sw_params/1e6:.3f}M)")
    print(f"  MatchedTransformer: {mt_params:,} params  ({mt_params/1e6:.3f}M)")
    print(f"  Delta             : {abs(sw_params - mt_params):,} params "
          f"({abs(sw_params-mt_params)/sw_params*100:.2f}%)")
    print(f"  Transformer arch  : dim=128, heads=4, ff_mult=4, 11 layers, "
          f"explicit causal mask")
    print(f"  Batch=1 for all measurements  (throughput = seq_len / fwd_time_s)")
    print()

    # ── Sweep ─────────────────────────────────────────────────────
    print(f"  {'seq':>6}  {'SW mem MB':>10}  {'MT mem MB':>10}  "
          f"{'MT/SW mem':>9}  "
          f"{'SW tok/s':>9}  {'MT tok/s':>9}  {'MT/SW tput':>10}")
    print("  " + "─" * 70)

    results = []
    for T in SEQ_LENS:
        # Spider Web
        try:
            sw_mb, sw_tps = measure(sw_fn, T, device, ctx, N_WARMUP, N_RUNS)
            sw_mb_s  = f"{sw_mb:10.1f}"
            sw_tps_s = f"{sw_tps:9.0f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sw_mb, sw_tps = float("nan"), float("nan")
            sw_mb_s = f"{'OOM':>10}"; sw_tps_s = f"{'OOM':>9}"

        # Matched transformer
        try:
            mt_mb, mt_tps = measure(mt, T, device, ctx, N_WARMUP, N_RUNS)
            mt_mb_s  = f"{mt_mb:10.1f}"
            mt_tps_s = f"{mt_tps:9.0f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mt_mb, mt_tps = float("nan"), float("nan")
            mt_mb_s = f"{'OOM':>10}"; mt_tps_s = f"{'OOM':>9}"

        try:
            mem_ratio = f"{mt_mb / sw_mb:9.2f}×"
        except Exception:
            mem_ratio = f"{'—':>9}"
        try:
            tput_ratio = f"{mt_tps / sw_tps:10.1f}×"
        except Exception:
            tput_ratio = f"{'—':>10}"

        print(f"  {T:>6}  {sw_mb_s}  {mt_mb_s}  {mem_ratio}  "
              f"{sw_tps_s}  {mt_tps_s}  {tput_ratio}")

        results.append((T, sw_mb, mt_mb, sw_tps, mt_tps))

    # ── Summary ───────────────────────────────────────────────────
    print()
    valid = [(T, sm, mm, st, mt) for T, sm, mm, st, mt in results
             if sm == sm and mm == mm]   # drop NaN

    if valid:
        T_min, sw_mb0, mt_mb0, *_ = valid[0]
        T_max, sw_mbN, mt_mbN, sw_tpsN, mt_tpsN = valid[-1]

        # memory growth ratios
        if sw_mb0 > 0 and mt_mb0 > 0:
            sw_growth = sw_mbN / sw_mb0
            mt_growth = mt_mbN / mt_mb0
            print(f"  Memory growth ({T_min}→{T_max} tokens):")
            print(f"    Spider Web       : {sw_growth:.1f}×")
            print(f"    Matched Transformer: {mt_growth:.1f}×")

        # crossover: first seq_len where SW uses LESS memory than MT
        crossover = None
        for T, sm, mm, *_ in valid:
            if sm < mm:
                crossover = T
                break
        if crossover:
            print(f"  Memory crossover: SW < MT at seq_len = {crossover}")
        else:
            print(f"  Memory crossover: SW never uses less memory than MT "
                  f"in the tested range (SW overhead dominates up to {T_max})")

        print(f"  Throughput at seq_len={T_max}: "
              f"SW {sw_tpsN:.0f} tok/s  MT {mt_tpsN:.0f} tok/s  "
              f"(MT is {mt_tpsN/sw_tpsN:.1f}× faster)")

    # ── Test 3 note ───────────────────────────────────────────────
    print()
    print("━" * 72)
    print("Test 3 — CE vs Published TinyStories Baselines")
    print("━" * 72)
    print()
    print("  Spider Web SLM CE (50-batch eval, 5K SentencePiece vocab): 4.77")
    print("  Random baseline (log 5000):                                8.52")
    print("  Reduction from random:                      44.3%")
    print()
    print("  WHY DIRECT COMPARISON IS NOT VALID:")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Eldan & Li 2023 (TinyStories paper) uses a 10K-token GPT-2-derived")
    print("  vocabulary. Spider Web uses a 5K SentencePiece vocabulary.")
    print("  Cross-entropy is tokenizer-relative: a lower vocab makes CE lower")
    print("  by construction (fewer options → lower entropy of the true distribution).")
    print()
    print("  Random baseline:  log(5000)  = 8.517  (Spider Web tokenizer)")
    print("  Random baseline:  log(10000) = 9.210  (TinyStories paper tokenizer)")
    print()
    print("  Additionally, the TinyStories paper does not report CE/perplexity")
    print("  in a table — its primary evaluation is GPT-4 quality scoring on")
    print("  grammar, creativity, and consistency. Numeric loss is only shown")
    print("  as learning curves in a figure (not tabulated).")
    print()
    print("  WHAT CAN BE SAID:")
    print("  Spider Web achieves a 44% reduction from random-baseline CE on the")
    print("  TinyStories training distribution, at 3.48M parameters and 20,000")
    print("  training steps on a single consumer GPU. Community reimplementations")
    print("  of TinyStories with comparable budgets (e.g., the Stanford CS224N")
    print("  curriculum-learning project) report GPT-Neo 1M-param models reaching")
    print("  CE ~2.0–2.5 on a 10K-vocab tokenizer after full training — a much")
    print("  larger budget. Comparable-budget experiments by the community suggest")
    print("  that a same-budget standard transformer would outperform Spider Web's")
    print("  current CE, consistent with the preliminary ablation finding at step 3600.")
    print()
    print("━" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
