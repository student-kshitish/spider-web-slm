# probe.py — capability probes, inference only

import torch
import torch.nn.functional as F
import sentencepiece as spm
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb
from infer import sample_token

CKPT = "checkpoints/final_run/best.pt"
SEQ  = 128
TAU  = 0.1
REPEAT_PENALTY = 1.3
REPEAT_WINDOW  = 32

torch.manual_seed(42)


def load():
    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = SEQ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpiderWeb(cfg).to(device)
    ckpt   = torch.load(CKPT, map_location=device)
    state  = {k: v.float() if v.is_floating_point() else v
               for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")
    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    return model, sp, device, ctx


def top5(model, sp, device, ctx, prompt):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        out    = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, 5)
    return [(sp.DecodeIds([i.item()]), v.item()) for i, v in zip(idxs, vals)]


def cont(model, sp, device, ctx, prompt, n=15, temperature=0.7, seed=42):
    torch.manual_seed(seed)
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    with torch.no_grad():
        for _ in range(n):
            with ctx:
                o = model(inp[:, -SEQ:], tau=TAU, hard=True)
            logits = o["logits"][0, -1, :].float()
            recent = inp[0, -REPEAT_WINDOW:].tolist()
            for t in set(recent):
                logits[t] = logits[t] / REPEAT_PENALTY if logits[t] > 0 else logits[t] * REPEAT_PENALTY
            nxt = sample_token(logits, temperature, top_k=40, top_p=0.9) if temperature > 0 \
                  else int(logits.argmax().item())
            out_ids.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
    return sp.DecodeIds(ids + out_ids)


def show_top5(label, tokens):
    print(f"  Prompt: \"{label}\"")
    for rank, (tok, prob) in enumerate(tokens, 1):
        bar = "█" * int(prob * 40)
        print(f"    {rank}. {tok!r:<18} {prob:.4f}  {bar}")


def section(title):
    print(f"\n{'━'*62}")
    print(f"  {title}")
    print(f"{'━'*62}")


def verdict(label):
    print(f"  [{label}]")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, sp, device, ctx = load()
    print(f"Model: {CKPT}  dim=64  best_loss=4.4212\n")

    # ══════════════════════════════════════════════════════════
    section("GROUP A — Syntax & completion  (in range)")
    section_prompts = [
        "The sky is",
        "The grass is",
        "She opened the",
        "He ran very",
    ]
    for p in section_prompts:
        show_top5(p, top5(model, sp, device, ctx, p))
        print()

    # ══════════════════════════════════════════════════════════
    section("GROUP B — Single-entity tracking  (edge of range)")

    p1 = "Lily had a red ball. She threw the"
    t1 = top5(model, sp, device, ctx, p1)
    c1 = cont(model, sp, device, ctx, p1, n=20)
    print(f'  Prompt: "{p1}"')
    print(f"  Continuation (20 tok): {c1}")
    print(f"  Top-5 next tokens:")
    for rank, (tok, prob) in enumerate(t1, 1):
        print(f"    {rank}. {tok!r:<18} {prob:.4f}")
    if t1[0][0].strip().lower() in ("ball", "ball."):
        verdict("PASS — 'ball' is top-1")
    elif any(tok.strip().lower().startswith("ball") for tok, _ in t1):
        verdict("PARTIAL — 'ball' in top-5 but not top-1")
    else:
        verdict("FAIL — 'ball' not in top-5")
    print()

    p2 = "Tom was sad. Tom started to"
    t2 = top5(model, sp, device, ctx, p2)
    c2 = cont(model, sp, device, ctx, p2, n=20)
    print(f'  Prompt: "{p2}"')
    print(f"  Continuation (20 tok): {c2}")
    print(f"  Top-5 next tokens:")
    for rank, (tok, prob) in enumerate(t2, 1):
        print(f"    {rank}. {tok!r:<18} {prob:.4f}")
    emotion_sad = {"cry", "cri", "weep", "sob", "feel", "frown", "sigh"}
    top1 = t2[0][0].strip().lower()
    if any(e in top1 for e in emotion_sad):
        verdict("PASS — emotionally consistent top-1")
    elif any(any(e in tok.strip().lower() for e in emotion_sad) for tok, _ in t2):
        verdict("PARTIAL — sad-consistent token in top-5")
    else:
        verdict(f"FAIL — top-1 is {t2[0][0]!r}, not emotionally consistent")
    print()

    # ══════════════════════════════════════════════════════════
    section("GROUP C — Simple cause-effect  (edge of range)")

    p3 = "It was raining, so she took her"
    t3 = top5(model, sp, device, ctx, p3)
    print(f'  Prompt: "{p3}"')
    print(f"  Top-5 next tokens:")
    for rank, (tok, prob) in enumerate(t3, 1):
        print(f"    {rank}. {tok!r:<18} {prob:.4f}")
    causal_ok = {"umbrella", "coat", "jacket", "raincoat", "hood", "hat", "bag"}
    if any(tok.strip().lower() in causal_ok for tok, _ in t3[:1]):
        verdict("PASS — causally correct top-1")
    elif any(tok.strip().lower() in causal_ok for tok, _ in t3):
        verdict(f"PARTIAL — causal token in top-5")
    else:
        verdict(f"FAIL — top-1 is {t3[0][0]!r}, not causally grounded")
    print()

    p4 = "He was hungry, so he ate a"
    t4 = top5(model, sp, device, ctx, p4)
    print(f'  Prompt: "{p4}"')
    print(f"  Top-5 next tokens:")
    for rank, (tok, prob) in enumerate(t4, 1):
        print(f"    {rank}. {tok!r:<18} {prob:.4f}")
    food = {"sandwich", "cookie", "apple", "banana", "cake", "big",
            "piece", "bowl", "lot", "snack", "meal", "bite", "fruit",
            "pizza", "bread", "biscuit", "cracker", "little"}
    if any(tok.strip().lower() in food for tok, _ in t4[:1]):
        verdict("PASS — food-consistent top-1")
    elif any(tok.strip().lower() in food for tok, _ in t4):
        verdict("PARTIAL — food token in top-5")
    else:
        verdict(f"FAIL — top-1 is {t4[0][0]!r}")
    print()

    # ══════════════════════════════════════════════════════════
    section("GROUP D — Reasoning / relations  (expected: out of range)")
    print("  (Expected failures — observing HOW the model fails)")
    print()

    reasoning_prompts = [
        ("Sara is taller than Ben. Who is shorter?",
         30, "should answer 'Ben' — will it?"),
        ("There are 2 cats and 1 dog. How many animals?",
         20, "should answer '3' — will it?"),
        ("The ball is in the box. The box is on the table. Where is the ball?",
         30, "should answer 'in the box' or 'on the table' — will it?"),
    ]

    for prompt, n_tok, note in reasoning_prompts:
        c = cont(model, sp, device, ctx, prompt, n=n_tok, temperature=0.7)
        t = top5(model, sp, device, ctx, prompt)
        print(f'  Prompt: "{prompt}"')
        print(f"  Note  : {note}")
        print(f"  Cont  : {c}")
        print(f"  Top-5 :")
        for rank, (tok, prob) in enumerate(t, 1):
            print(f"    {rank}. {tok!r:<18} {prob:.4f}")
        print()
