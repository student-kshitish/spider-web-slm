# quick_infer.py — four capability tests, inference only, no training

import torch
import torch.nn.functional as F
import sentencepiece as spm

from config import get_config
from core.web import SpiderWeb
from infer import top_k_filter, top_p_filter, sample_token

CKPT   = "checkpoints/final_run/best.pt"
DIM    = 64
HIDDEN = 256
SEQ    = 128
TAU    = 0.1          # routing temperature at inference
REPEAT_PENALTY = 1.3
REPEAT_WINDOW  = 32

torch.manual_seed(42)


def load():
    cfg = get_config()
    cfg.model.dim        = DIM
    cfg.model.hidden_dim = HIDDEN
    cfg.model.max_seq_len = SEQ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpiderWeb(cfg).to(device)

    ckpt  = torch.load(CKPT, map_location=device)
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")

    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else __import__("contextlib").nullcontext())

    return model, sp, cfg, device, ctx


def gen(model, sp, cfg, device, ctx,
        prompt, max_new=60, temperature=0.7, top_k=40, top_p=0.9,
        seed=42):
    torch.manual_seed(seed)
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    with torch.no_grad():
        for _ in range(max_new):
            x = inp[:, -SEQ:]
            with ctx:
                o = model(x, tau=TAU, hard=True)
            logits = o["logits"][0, -1, :].float()
            recent = inp[0, -REPEAT_WINDOW:].tolist()
            for t in set(recent):
                logits[t] = logits[t] / REPEAT_PENALTY if logits[t] > 0 else logits[t] * REPEAT_PENALTY
            nxt = sample_token(logits, temperature, top_k, top_p)
            out_ids.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
    return sp.DecodeIds(ids + out_ids)


# ── TEST 1: Generation across prompts ─────────────────────────────────────
def test1(model, sp, cfg, device, ctx):
    print("━" * 60)
    print("TEST 1 — Generation (60 tokens, temp=0.7, top_k=40, top_p=0.9)")
    print("━" * 60)
    prompts = [
        "Once upon a time",
        "The cat sat on the",
        "Tom was very happy because",
        "One day a little girl found a",
        "The end",
    ]
    for p in prompts:
        out = gen(model, sp, cfg, device, ctx, p, max_new=60, temperature=0.7)
        print(f'\n["{p}"]\n  {out}')
    print()


# ── TEST 2: Greedy vs sampled ──────────────────────────────────────────────
def test2(model, sp, cfg, device, ctx):
    print("━" * 60)
    print("TEST 2 — Greedy vs Sampled  (prompt: \"Lily wanted to\")")
    print("━" * 60)
    prompt = "Lily wanted to"

    # greedy: temperature → 0 (argmax)
    torch.manual_seed(42)
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    greedy_ids = []
    with torch.no_grad():
        for _ in range(60):
            x = inp[:, -SEQ:]
            with ctx:
                o = model(x, tau=TAU, hard=True)
            logits = o["logits"][0, -1, :].float()
            nxt = int(logits.argmax().item())
            greedy_ids.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
    greedy_text = sp.DecodeIds(ids + greedy_ids)

    # sampled: temp=0.8
    sampled_text = gen(model, sp, cfg, device, ctx, prompt,
                       max_new=60, temperature=0.8, seed=7)

    print(f'\n  GREEDY  : {greedy_text}')
    print(f'\n  SAMPLED : {sampled_text}')
    print()


# ── TEST 3: Top-10 next-token probabilities ────────────────────────────────
def test3(model, sp, cfg, device, ctx):
    print("━" * 60)
    print('TEST 3 — Top-10 next tokens')
    print('  prompt: "Once upon a time, there was a little"')
    print("━" * 60)
    prompt = "Once upon a time, there was a little"
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with ctx:
            o = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = o["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    top_p_vals, top_p_idx = torch.topk(probs, 10)
    print(f"\n  {'Rank':<5} {'Token':<20} {'Prob':>8}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*8}")
    for rank, (prob, idx) in enumerate(zip(top_p_vals.tolist(), top_p_idx.tolist()), 1):
        tok = repr(sp.DecodeIds([idx]))
        print(f"  {rank:<5} {tok:<20} {prob:>8.4f}")
    print()


# ── TEST 4: Short vs long coherence ───────────────────────────────────────
def test4(model, sp, cfg, device, ctx):
    print("━" * 60)
    print('TEST 4 — Coherence: 20 vs 100 tokens  (prompt: "Once upon a time")')
    print("━" * 60)
    prompt = "Once upon a time"
    short = gen(model, sp, cfg, device, ctx, prompt, max_new=20,  temperature=0.7, seed=42)
    long  = gen(model, sp, cfg, device, ctx, prompt, max_new=100, temperature=0.7, seed=42)
    print(f'\n  20 tokens : {short}')
    print(f'\n  100 tokens: {long}')
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, sp, cfg, device, ctx = load()
    print(f"Loaded {CKPT}  |  device={device}  |  dim={DIM}\n")

    test1(model, sp, cfg, device, ctx)
    test2(model, sp, cfg, device, ctx)
    test3(model, sp, cfg, device, ctx)
    test4(model, sp, cfg, device, ctx)

    print("━" * 60)
    print("Done.")
