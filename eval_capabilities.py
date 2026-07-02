# eval_capabilities.py — six-category capability eval for any Spider Web checkpoint

import argparse
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from infer import sample_token
import sentencepiece as spm

SEQ, TAU = 128, 0.1
REPEAT_PENALTY, REPEAT_WINDOW = 1.3, 32


def load(ckpt, dim=64, hidden=256):
    cfg = get_config()
    cfg.model.dim = dim
    cfg.model.hidden_dim = hidden
    cfg.model.max_seq_len = SEQ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpiderWeb(cfg).to(device)
    ck = torch.load(ckpt, map_location=device)
    state = ck.get("model", ck)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    model.eval()
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")
    ctx = torch.autocast("cuda", torch.bfloat16) if device.type == "cuda" else nullcontext()
    has_recall = any("recall_proj" in k for k in state)
    return model, sp, device, ctx, has_recall, missing


def top5(model, sp, device, ctx, prompt):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        out = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, 5)
    return [(sp.DecodeIds([i.item()]), v.item()) for i, v in zip(idxs, vals)]


def cont(model, sp, device, ctx, prompt, n=20, temperature=0.7, seed=42):
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
            nxt = (
                sample_token(logits, temperature, top_k=40, top_p=0.9)
                if temperature > 0
                else int(logits.argmax().item())
            )
            out_ids.append(nxt)
            inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)
    return sp.DecodeIds(ids + out_ids)


def score_recall(t5, ok_set, target):
    top1 = t5[0][0].strip().lower().rstrip(".")
    in_top5 = any(
        tok.strip().lower().rstrip(".") in ok_set or target in tok.strip().lower()
        for tok, _ in t5
    )
    if top1 == target or target in top1:
        return "PASS", t5[0][0]
    if in_top5:
        return "PARTIAL", t5[0][0]
    return "FAIL", t5[0][0]


def score_set(t5, ok):
    top1 = t5[0][0].strip().lower()
    if top1 in ok:
        return "PASS", t5[0][0]
    if any(tok.strip().lower() in ok for tok, _ in t5):
        return "PARTIAL", t5[0][0]
    return "FAIL", t5[0][0]


def run_eval(ckpt, label=None):
    label = label or ckpt
    model, sp, device, ctx, has_recall, missing = load(ckpt)
    print(f"\n{'='*64}")
    print(f"  CAPABILITY EVAL — {label}")
    print(f"  Device: {device}  recall_proj trained: {has_recall}")
    if missing:
        print(f"  Missing keys (random init): {missing}")
    print(f"{'='*64}")

    results = {}

    # 1. Cross-sentence memory recall
    print("\n── 1. Cross-Sentence Memory Recall ──")
    recall_tests = [
        ("Lily had a red ball. She went to the park. Later she picked up the",
         {"ball", "ball.", "red", "it"}, "ball"),
        ("Tom found a blue hat. He put it on his head. Then Tom lost the",
         {"hat", "hat.", "blue", "it", "his"}, "hat"),
        ("The dog chased a cat. The cat ran away. The dog looked for the",
         {"cat", "cat.", "it"}, "cat"),
        ("Anna baked a cake. Her friends came over. They ate the",
         {"cake", "cake.", "it", "food"}, "cake"),
    ]
    recall_scores = []
    for prompt, ok, target in recall_tests:
        t5 = top5(model, sp, device, ctx, prompt)
        v, top = score_recall(t5, ok, target)
        recall_scores.append(v)
        print(f"  [{v}] {prompt[:55]}… → top-1={top!r}")

    # 1b. Recall loss (training signal)
    print("\n── 1b. Recall Loss Evaluation ──")
    model.train()
    loss_fn = SpiderWebLoss()
    x = torch.randint(0, 5000, (4, 128), device=device)
    y = torch.randint(0, 5000, (4, 128), device=device)
    with ctx:
        out = model(x, tau=1.0, hard=False)
        out["logits"] = out["logits"].float()
        _, m = loss_fn(out, y, w_recall=0.01, w_depth=0.02)
    model.eval()
    print(f"  recall_loss={m['recall']:.4f}  depth_loss={m['depth']:.4f}  ce={m['ce']:.4f}")
    recall_loss_ok = m["recall"] < 0.95 and has_recall
    results["recall_loss"] = "PASS" if recall_loss_ok else ("PARTIAL" if m["recall"] < 1.0 else "FAIL")

    # Ring stats snapshot
    with torch.no_grad(), ctx:
        rs = model(x[:2], tau=0.5, hard=False, return_ring_stats=True)["ring_stats"]
    NR = len(rs)
    inner_frac = sum(rs[r]["visit_frac"] for r in range(NR // 2)) * 100
    r0_visits = rs[0]["visits"]
    print(f"  Ring 0 visits={r0_visits}  inner-ring fraction={inner_frac:.1f}%")
    results["ring_mechanism"] = "PASS" if r0_visits > 0 and inner_frac > 30 else "FAIL"

    # 2. Causal reasoning
    print("\n── 2. Causal Reasoning ──")
    causal_tests = [
        ("It was raining, so she took her", {"umbrella", "coat", "jacket", "raincoat", "hood", "hat", "bag"}),
        ("He was hungry, so he ate a", {"sandwich", "cookie", "apple", "banana", "cake", "big", "piece", "bowl", "snack", "meal", "pizza", "bread", "fruit", "little"}),
        ("The sun was hot, so he drank some", {"water", "juice", "cold", "ice", "lemonade", "milk", "soda", "drink"}),
        ("She was tired, so she went to", {"bed", "sleep", "rest", "her", "the", "home", "nap"}),
    ]
    causal_scores = []
    for prompt, ok in causal_tests:
        t5 = top5(model, sp, device, ctx, prompt)
        v, top = score_set(t5, ok)
        causal_scores.append(v)
        print(f"  [{v}] {prompt!r} → top-1={top!r}")

    # 3. Relational reasoning
    print("\n── 3. Relational Reasoning ──")
    rel_tests = [
        ("Sara is taller than Ben. Who is shorter?", {"ben", "ben."}, 25),
        ("The ball is in the box. The box is on the table. Where is the ball?",
         {"table", "box", "in", "on"}, 30),
        ("Tom is older than Lily. Lily is older than Max. Who is youngest?",
         {"max", "max."}, 25),
    ]
    rel_scores = []
    for prompt, ok, n in rel_tests:
        c = cont(model, sp, device, ctx, prompt, n=n)
        t5 = top5(model, sp, device, ctx, prompt)
        ans = c[len(prompt):].strip().lower()[:50]
        if any(w in ans for w in ok):
            v = "PASS"
        elif any(tok.strip().lower() in ok for tok, _ in t5):
            v = "PARTIAL"
        else:
            v = "FAIL"
        rel_scores.append(v)
        print(f"  [{v}] {prompt[:50]}… → cont={ans!r}")

    # 4. Numerical reasoning
    print("\n── 4. Numerical Reasoning ──")
    num_tests = [
        ("There are 2 cats and 1 dog. How many animals?", {"3", "three"}),
        ("She had three apples. She ate one. Now she has", {"two", "2"}),
        ("Tom has 5 toys. He gave 2 away. Tom has", {"3", "three"}),
        ("One plus one equals", {"two", "2"}),
    ]
    num_scores = []
    for prompt, ok in num_tests:
        t5 = top5(model, sp, device, ctx, prompt)
        c = cont(model, sp, device, ctx, prompt, n=10)
        ans = c[len(prompt):].strip().lower()
        if any(w in t5[0][0].strip().lower() for w in ok):
            v = "PASS"
        elif any(tok.strip().lower() in ok for tok, _ in t5) or any(w in ans.split()[:3] for w in ok):
            v = "PARTIAL"
        else:
            v = "FAIL"
        num_scores.append(v)
        print(f"  [{v}] {prompt[:45]}… → top-1={t5[0][0]!r}")

    # 5. Contrast & semantic relations
    print("\n── 5. Contrast & Semantic Relations ──")
    contrast_tests = [
        ("The dog was big and the cat was", {"small", "little", "tiny", "not", "short", "thin", "also"}),
        ("The sky is blue but the grass is", {"green", "brown", "not", "yellow", "dark"}),
        ("She was happy, not", {"sad", "angry", "unhappy", "mad", "upset", "scared"}),
        ("Hot is the opposite of", {"cold", "cool", "freezing", "warm"}),
    ]
    contrast_scores = []
    for prompt, ok in contrast_tests:
        t5 = top5(model, sp, device, ctx, prompt)
        v, top = score_set(t5, ok)
        contrast_scores.append(v)
        print(f"  [{v}] {prompt!r} → top-1={top!r}")

    # 6. Generalization
    print("\n── 6. Generalization (OOD) ──")
    gen_tests = [
        ("The astronaut floated in", "genre OOD"),
        ("The robot picked up the", "entity OOD"),
        ("In the year 3000, people will", "temporal OOD"),
    ]
    gen_scores = []
    for prompt, note in gen_tests:
        t5 = top5(model, sp, device, ctx, prompt)
        c = cont(model, sp, device, ctx, prompt, n=15)
        cont_text = c[len(prompt):].strip()
        p = t5[0][1]
        if len(cont_text) > 3 and p > 0.01:
            v = "PASS"
        elif p > 0.005:
            v = "PARTIAL"
        else:
            v = "FAIL"
        gen_scores.append(v)
        print(f"  [{v}] [{note}] cont={cont_text[:45]!r}  p={p:.3f}")

    # Summary
    def summarize(scores):
        if all(s == "FAIL" for s in scores):
            return "FAIL"
        if any(s == "PASS" for s in scores):
            return "PARTIAL" if any(s == "FAIL" for s in scores) else "PASS"
        return "PARTIAL"

    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    summary = {
        "Cross-Sentence Recall": summarize(recall_scores),
        "Recall Loss Signal": results["recall_loss"],
        "Ring Mechanism (Ring 0 alive)": results["ring_mechanism"],
        "Causal Reasoning": summarize(causal_scores),
        "Relational Reasoning": summarize(rel_scores),
        "Numerical Reasoning": summarize(num_scores),
        "Contrast & Semantic": summarize(contrast_scores),
        "Generalization": summarize(gen_scores),
    }
    for k, v in summary.items():
        print(f"  {k:<32} {v}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/radial_probe/best.pt")
    p.add_argument("--label", default=None)
    p.add_argument("--compare", action="store_true", help="also eval final_run/best.pt")
    args = p.parse_args()
    run_eval(args.ckpt, args.label or args.ckpt)
    if args.compare:
        run_eval("checkpoints/final_run/best.pt", "final_run/best.pt (baseline, no radial training)")
