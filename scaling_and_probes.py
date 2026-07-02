# scaling_and_probes.py
# Part 1: memory/compute scaling sweep (Spider Web vs standard self-attention)
# Part 2: four more capability probes

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb
from infer import generate, _block_ngrams, sample_token, INFER_TAU

CKPT  = "checkpoints/final_run/best.pt"
VOCAB = 5000
DIM   = 64
SEQ_LENS = [64, 128, 256, 512, 1024, 2048]
N_WARMUP  = 2
N_RUNS    = 5   # averaged over this many forward passes per seq_len


# ══════════════════════════════════════════════════════════════════
# PART 1 — SCALING BENCHMARK
# ══════════════════════════════════════════════════════════════════

class AttentionBaseline(nn.Module):
    """
    Single-layer causal transformer. Same dim=64, same vocab=5000, same
    hidden ratio (×4 = 256).  Uses full O(T²) self-attention — our control.
    """
    def __init__(self, vocab=VOCAB, d=DIM, heads=4, ff_mult=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(d * ff_mult, d, bias=False),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.embed(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h2, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff(h))
        return h


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_forward(model_fn, seq_len, device, autocast_ctx, n_warmup, n_runs):
    """
    Returns (peak_activation_MB, avg_time_ms).
    peak_activation_MB subtracts the model's baseline allocation so only
    activation memory is reported (makes the growth curve seq-len-dependent only).
    """
    x = torch.randint(0, VOCAB, (1, seq_len), device=device)

    # baseline: model params already on GPU, no activations yet
    torch.cuda.synchronize(device)
    baseline_bytes = torch.cuda.memory_allocated(device)

    # warmup (fills caches, JIT, etc.)
    for _ in range(n_warmup):
        with torch.no_grad(), autocast_ctx:
            model_fn(x)
    torch.cuda.synchronize(device)

    # measure
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad(), autocast_ctx:
            model_fn(x)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    peak_bytes  = torch.cuda.max_memory_allocated(device)
    activation_mb = (peak_bytes - baseline_bytes) / 1024 / 1024
    time_ms     = (t1 - t0) / n_runs * 1000
    return max(activation_mb, 0.0), time_ms


def run_scaling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    if not on_cuda:
        print("WARNING: no CUDA — scaling test will be slow on CPU")

    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16) if on_cuda else nullcontext()
    )

    # ── Spider Web ───────────────────────────────────────────────
    cfg = get_config()
    cfg.model.dim         = DIM
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = max(SEQ_LENS)   # allow long contexts

    sw = SpiderWeb(cfg).to(device)
    ckpt  = torch.load(CKPT, map_location=device)
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in ckpt["model"].items()}
    sw.load_state_dict(state)
    sw.eval()

    def sw_fn(x):
        out = sw(x, tau=INFER_TAU, hard=True)
        return out["logits"]

    # ── Attention baseline ────────────────────────────────────────
    attn = AttentionBaseline().to(device)
    if on_cuda:
        # keep in float32 master weights to match Spider Web
        pass
    attn.eval()

    sw_params   = count_params(sw)
    attn_params = count_params(attn)

    print("━" * 64)
    print("PART 1 — Memory / Compute Scaling")
    print("━" * 64)
    print(f"  Spider Web params  : {sw_params/1e6:.3f}M")
    print(f"  Attention baseline : {attn_params/1e6:.3f}M  "
          f"(single-layer, d={DIM}, h={DIM*4}, heads=4)")
    print(f"  Note: outputs past seq_len=128 are meaningless (model trained "
          f"to 128).")
    print(f"        Memory/compute scaling is the valid measurement.\n")

    hdr = f"  {'seq_len':>7}  {'SW mem MB':>10}  {'SW ms':>8}  "  \
          f"{'Attn mem MB':>11}  {'Attn ms':>8}  {'mem ratio':>10}"
    print(hdr)
    print("  " + "-" * 62)

    sw_rows   = []
    attn_rows = []

    for T in SEQ_LENS:
        # Spider Web
        try:
            sw_mb, sw_ms = measure_forward(sw_fn, T, device, autocast_ctx,
                                           N_WARMUP, N_RUNS)
            sw_tag = f"{sw_mb:10.1f}  {sw_ms:8.1f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sw_mb, sw_ms = float("nan"), float("nan")
            sw_tag = f"{'OOM':>10}  {'OOM':>8}"
        sw_rows.append((T, sw_mb, sw_ms))

        # Attention baseline
        try:
            at_mb, at_ms = measure_forward(attn, T, device, autocast_ctx,
                                           N_WARMUP, N_RUNS)
            at_tag = f"{at_mb:11.1f}  {at_ms:8.1f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            at_mb, at_ms = float("nan"), float("nan")
            at_tag = f"{'OOM':>11}  {'OOM':>8}"
        attn_rows.append((T, at_mb, at_ms))

        # ratio
        try:
            ratio = f"{at_mb / sw_mb:.2f}×"
        except Exception:
            ratio = "—"

        print(f"  {T:>7}  {sw_tag}  {at_tag}  {ratio:>10}")

    # summary at largest non-OOM pair
    print()
    valid = [(s, a) for s, a in zip(sw_rows, attn_rows)
             if not (s[1] != s[1]) and not (a[1] != a[1])]   # filter NaN
    if valid:
        T_max = valid[-1][0][0]
        sw_peak = valid[-1][0][1]
        at_peak = valid[-1][1][1]
        print(f"  At seq_len={T_max}: Spider Web {sw_peak:.1f} MB  "
              f"vs Attention {at_peak:.1f} MB  "
              f"({at_peak/sw_peak:.1f}× more for attention)")

    del sw, attn
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════
# PART 2 — CAPABILITY PROBES
# ══════════════════════════════════════════════════════════════════

GEN_KWARGS = dict(
    max_new_tokens=80, temperature=0.6, top_k=30, top_p=1.0,
    no_repeat_ngram_size=3, stop_at_sentence=True, min_new_tokens=12,
)
SEQ = 128


def load_for_infer(device):
    cfg = get_config()
    cfg.model.dim, cfg.model.hidden_dim, cfg.model.max_seq_len = DIM, 256, SEQ
    model = SpiderWeb(cfg).to(device)
    ckpt  = torch.load(CKPT, map_location=device)
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")
    ctx = (torch.autocast("cuda", torch.bfloat16) if device.type == "cuda"
           else nullcontext())
    return model, sp, ctx


def top5(model, sp, device, ctx, prompt):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        out    = model(inp[:, -SEQ:], tau=INFER_TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, 5)
    return [(sp.DecodeIds([i.item()]), v.item())
            for i, v in zip(idxs, vals)]


def gen(model, sp, device, prompt, seed=42):
    torch.manual_seed(seed)
    return generate(model=model, sp=sp, prompt=prompt,
                    seq_len=SEQ, device=device, **GEN_KWARGS)


def run_probes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp, ctx = load_for_infer(device)

    print("\n" + "━" * 64)
    print("PART 2 — Capability Probes  "
          "(temp=0.6, top_k=30, ngram=3, stop_at_sentence)")
    print("━" * 64)

    probes = [
        (
            "The dog was big and the cat was",
            "contrast / comparison within one sentence",
            # success: adjective like 'small', 'little', 'tiny', 'not', 'also big'
            lambda t5: (
                "PASS"   if any(tok.strip().lower() in
                                {"small","little","tiny","not","also","short","thin"}
                                for tok, _ in t5[:2])
                else "PARTIAL" if any(tok.strip().lower() in
                                      {"small","little","tiny","not","also","short","thin"}
                                      for tok, _ in t5)
                else "FAIL"
            ),
        ),
        (
            "She had three apples. She ate one. Now she has",
            "simple counting (likely fails)",
            lambda t5: (
                "PASS"   if any(tok.strip() in {"two","2","three","3"}
                                for tok, _ in t5[:1])
                else "PARTIAL" if any(tok.strip() in {"two","2","three","3","one","1"}
                                      for tok, _ in t5)
                else "FAIL"
            ),
        ),
        (
            "Once there was a princess who lived in a",
            "genre / setting completion (likely works)",
            lambda t5: (
                "PASS"   if any(tok.strip().lower() in
                                {"castle","tower","forest","kingdom","big","beautiful",
                                 "far","land","little","magical","enchanted","palace"}
                                for tok, _ in t5[:2])
                else "PARTIAL" if any(tok.strip().lower() in
                                      {"castle","tower","forest","kingdom","big",
                                       "beautiful","far","land","little","magical",
                                       "enchanted","palace"}
                                      for tok, _ in t5)
                else "FAIL"
            ),
        ),
        (
            "He said hello and she said",
            "dialogue turn-taking",
            lambda t5: (
                "PASS"   if any(tok.strip().lower() in
                                {'"', "'", "hello", "hi", "yes", "no",
                                 ",", "goodbye", "thank"}
                                for tok, _ in t5[:2])
                else "PARTIAL" if any(tok.strip().lower() in
                                      {'"', "'", "hello", "hi", "yes", "no",
                                       "goodbye", "thank", "okay", "sorry"}
                                      for tok, _ in t5)
                else "FAIL"
            ),
        ),
    ]

    for prompt, note, judge in probes:
        t5   = top5(model, sp, device, ctx, prompt)
        cont = gen(model, sp, device, prompt)
        verdict = judge(t5)
        print(f'\n  Prompt : "{prompt}"')
        print(f"  Note   : {note}")
        print(f"  Cont   : {cont}")
        print(f"  Top-5 next tokens:")
        for rank, (tok, prob) in enumerate(t5, 1):
            print(f"    {rank}. {tok!r:<18} {prob:.4f}")
        print(f"  Verdict: {verdict}")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_scaling()
    run_probes()
    print("\n" + "━" * 64)
    print("Done.")
