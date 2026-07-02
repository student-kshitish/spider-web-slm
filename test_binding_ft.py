"""
Step 3 test — side-by-side A/B of the fine-tuned arms.

  Arm A (on)  : checkpoints/binding_ft_on/best.pt   (w_depth=0.005, w_recall=0.02)
  Arm B (off) : checkpoints/binding_ft_off/best.pt  (control, weights 0)

Both start from the same competent baseline, train on the same data with the
same seed. The only difference is the mechanism. So any difference in binding
is attributable to the mechanism.

Probes (top-5 next token, shown for BOTH arms):
  1. "Lily had a red ball. She threw the"          -> is "ball"/"red" in top-5?
  2. "Sara had a blue hat. Later, Sara wore the"    -> is "hat"/"blue" in top-5?
  3. in-distribution competence check (predict a noun after "the red ___")
"""

import torch
import torch.nn.functional as F
import sentencepiece as spm
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb

SEQ = 256
TAU = 0.1

ARMS = [
    ("A:ON ", "checkpoints/binding_ft_on/best.pt"),
    ("B:OFF", "checkpoints/binding_ft_off/best.pt"),
]

# (prompt, wanted-substrings, is_competence_check)
PROBES = [
    ("Lily had a red ball. She threw the",           {"ball", "red"},  False),
    ("Sara had a blue hat. Later, Sara wore the",     {"hat", "blue"},  False),
    # in-distribution training format: after "the red" a NOUN should follow
    ("Lily had a red ball. she went to the park. Then Lily threw the red",
     {"ball", "dog", "cat", "hat", "doll", "box", "kite", "cup", "car",
      "fish", "frog", "cake", "key", "drum", "bell", "duck", "bear", "boat",
      "book", "shoe"}, True),
]

torch.manual_seed(42)


def load(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = {k: (v.float() if v.is_floating_point() else v)
              for k, v in ckpt["model"].items()}
    cfg = get_config()
    cfg.model.dim = 64; cfg.model.hidden_dim = 256; cfg.model.max_seq_len = SEQ
    cfg.memory.slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    model = SpiderWeb(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    return model, device, ctx, ckpt.get("step", "?"), ckpt.get("ema_ce", None)


def topk(model, sp, device, ctx, prompt, k=5):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        logits = model(inp[:, -SEQ:], tau=TAU, hard=True)["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(sp.DecodeIds([i.item()]).strip(), v.item()) for i, v in zip(idxs, vals)]


def hit(tokens, wanted):
    for tok, _ in tokens:
        t = tok.lower()
        for w in wanted:
            if w and w in t:
                return tok
    return None


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")

    # load both arms
    arms = {}
    for name, path in ARMS:
        try:
            arms[name] = load(path)
        except FileNotFoundError:
            print(f"!! {name}: checkpoint missing ({path})"); return
    for name, path in ARMS:
        _, _, _, step, ema = arms[name]
        ema_s = f"{ema:.3f}" if isinstance(ema, float) else str(ema)
        print(f"{name}: {path}  step={step}  ema_ce={ema_s}")

    summary = []
    for prompt, wanted, is_comp in PROBES:
        print(f"\n{'='*70}")
        tag = "COMPETENCE CHECK (noun after 'the red ___')" if is_comp else "BINDING PROBE"
        print(f"{tag}")
        print(f'Prompt: "{prompt}"')
        print(f"want one of: {sorted(w for w in wanted)[:8]}{' ...' if len(wanted)>8 else ''}")
        print(f"{'-'*70}")
        row = {"prompt": prompt, "is_comp": is_comp}
        for name, _ in ARMS:
            model, device, ctx, _, _ = arms[name]
            t5 = topk(model, sp, device, ctx, prompt)
            h  = hit(t5, wanted)
            row[name] = h
            toks = "  ".join(f"{tok!r}:{prob:.3f}" for tok, prob in t5)
            mark = f"  => {'HIT '+repr(h) if h else 'absent'}"
            print(f"  [{name}] {toks}")
            print(f"          {mark}")
        summary.append(row)

    # ── decisive comparison ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DECISIVE COMPARISON — recalled entity in top-5?")
    print(f"{'='*70}")
    print(f"{'Probe':<52}{'A:ON':>8}{'B:OFF':>8}")
    print(f"{'-'*68}")
    binding_rows = [r for r in summary if not r["is_comp"]]
    a_only = 0
    both = 0
    for r in binding_rows:
        a = r["A:ON "]; b = r["B:OFF"]
        if a and not b: a_only += 1
        if a and b: both += 1
        short = (r["prompt"][:49] + "...") if len(r["prompt"]) > 52 else r["prompt"]
        print(f"{short:<52}{(repr(a) if a else 'absent'):>8}{(repr(b) if b else 'absent'):>8}")

    comp = [r for r in summary if r["is_comp"]][0]
    print(f"\nCompetence check (both should hit a noun): "
          f"A:ON={'OK' if comp['A:ON '] else 'FAIL'}  "
          f"B:OFF={'OK' if comp['B:OFF'] else 'FAIL'}")

    print(f"\n{'-'*68}")
    if a_only > 0:
        print(f"✓ VERDICT: mechanism (A) recalled the entity on {a_only}/{len(binding_rows)} "
              f"probe(s) where the control (B) did not.")
        print("  -> the depth+recall mechanism improves cross-sentence binding.")
    elif both > 0 and both == len(binding_rows):
        print("~ VERDICT: both arms recall the entity equally.")
        print("  -> binding came from the data/fine-tune, not specifically the mechanism.")
    else:
        print("✗ VERDICT: neither arm recalls the entity in top-5.")
        print("  -> mechanism does not help binding even from a competent start +")
        print("     dependency-rich data (and the control confirms it isn't just undertraining).")


if __name__ == "__main__":
    main()
