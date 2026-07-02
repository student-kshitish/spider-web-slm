"""
Step 3 — Test cross-sentence binding, baseline vs binding-trained.

Runs the SAME probes that failed before, on:
  - BASELINE : checkpoints/final_run/best.pt   (the model probe.py reported on)
  - BINDING  : checkpoints/binding_run/best.pt  (this experiment)

The decisive question: does the recalled entity (ball / hat / blue) now appear
in the top-5 next tokens where it was absent in the baseline?

Probes:
  1. "Lily had a red ball. She threw the"      -> is "ball" in top-5?
  2. "Tom was sad. Tom started to"             -> sad-consistent continuation?
  3. "Sara had a blue hat. Later, Sara wore the" -> is "hat"/"blue" recalled?
"""

import torch
import torch.nn.functional as F
import sentencepiece as spm
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb

SEQ = 256
TAU = 0.1

BASELINE_CKPT = "checkpoints/final_run/best.pt"
BINDING_CKPT  = "checkpoints/binding_run/best.pt"

PROMPTS = [
    ("Lily had a red ball. She threw the",            {"ball", "red"}),
    ("Tom was sad. Tom started to",                   {"cry", "cri", "weep", "sob", "feel", "frown", "sigh", "tom"}),
    ("Sara had a blue hat. Later, Sara wore the",      {"hat", "blue"}),
]

torch.manual_seed(42)


def load(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = {k: v.float() if v.is_floating_point() else v
              for k, v in ckpt["model"].items()}

    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = SEQ
    # auto-detect memory slots from checkpoint (final_run=32, binding_run=16)
    seed_key = "rings.0.0.memory.m_t_seed"
    if seed_key in state:
        cfg.memory.slots = state[seed_key].shape[0]
    model  = SpiderWeb(cfg).to(device)
    # strict=False: baseline (final_run) predates recall_proj, which is used
    # only in the recall LOSS, never in the logits path — so it cannot affect
    # top-5 predictions either way.
    missing, _ = model.load_state_dict(state, strict=False)
    if missing and missing != ["recall_proj.weight"]:
        print(f"  [warn] missing keys: {missing}")
    model.eval()
    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    return model, device, ctx, ckpt.get("step", "?"), ckpt.get("ema_ce", None)


def topk(model, sp, device, ctx, prompt, k=5):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        out    = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(sp.DecodeIds([i.item()]).strip(), v.item()) for i, v in zip(idxs, vals)]


def hit(tokens, wanted):
    """True if any top-k token contains any wanted word (case-insensitive)."""
    for tok, _ in tokens:
        t = tok.lower()
        for w in wanted:
            if w in t:
                return tok
    return None


def run_model(name, ckpt_path, sp):
    try:
        model, device, ctx, step, ema = load(ckpt_path)
    except FileNotFoundError:
        print(f"\n### {name}: checkpoint not found ({ckpt_path}) — skipping")
        return None
    ema_s = f"{ema:.3f}" if isinstance(ema, float) else str(ema)
    print(f"\n{'='*64}")
    print(f"### {name}   ({ckpt_path})   step={step}  ema_ce={ema_s}")
    print(f"{'='*64}")
    results = {}
    for prompt, wanted in PROMPTS:
        t5 = topk(model, sp, device, ctx, prompt)
        h  = hit(t5, wanted)
        results[prompt] = (t5, h)
        print(f'\nPrompt: "{prompt}"')
        print(f"  want one of: {sorted(wanted)}")
        for rank, (tok, prob) in enumerate(t5, 1):
            mark = "  <-- HIT" if (h is not None and tok == h) else ""
            print(f"    {rank}. {tok!r:<16} {prob:.4f}{mark}")
        print(f"  => {'RECALLED: ' + repr(h) if h else 'ABSENT (not in top-5)'}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")

    base = run_model("BASELINE (final_run)", BASELINE_CKPT, sp)
    bind = run_model("BINDING  (binding_run)", BINDING_CKPT, sp)

    if base is None or bind is None:
        return

    print(f"\n{'='*64}")
    print("DIRECT COMPARISON — recalled entity in top-5?")
    print(f"{'='*64}")
    print(f"{'Prompt':<46} {'Baseline':>9} {'Binding':>9}")
    print(f"{'-'*66}")
    improved = 0
    for prompt, _ in PROMPTS:
        b_hit = base[prompt][1]
        n_hit = bind[prompt][1]
        b_s = repr(b_hit) if b_hit else "absent"
        n_s = repr(n_hit) if n_hit else "absent"
        if not b_hit and n_hit:
            improved += 1
            n_s += " ✓NEW"
        short = (prompt[:43] + "...") if len(prompt) > 46 else prompt
        print(f"{short:<46} {b_s:>9} {n_s:>9}")

    print(f"\n{'-'*66}")
    if improved > 0:
        print(f"✓ VERDICT: binding improved on {improved}/{len(PROMPTS)} probes "
              f"(entity now in top-5 where it was absent).")
        print("  -> mechanism improves cross-sentence binding; full run justified.")
    else:
        print("✗ VERDICT: no probe gained the recalled entity.")
        print("  -> binding did not transfer even with dependency-rich data;")
        print("     consistent with a capacity wall blocking it.")


if __name__ == "__main__":
    main()
