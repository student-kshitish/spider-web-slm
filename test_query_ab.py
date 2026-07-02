"""
Step 3 test — side-by-side A/B of the orbital query-read arms.

  Arm A (on)  : checkpoints/query_on/best.pt   (use_query_read=True)
  Arm B (off) : checkpoints/query_off/best.pt  (control, read disabled)

Same competent start, same data/seed; only the orbital read differs. So any
binding difference is attributable to the read.

Reports:
  - top-5 next token for BOTH arms on 3 binding probes + a competence check
  - the learned per-ring bias (did the model favour inner rings?)
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
    ("A:ON ", "checkpoints/query_on/best.pt",  True),
    ("B:OFF", "checkpoints/query_off/best.pt", False),
]

NOUNS = {"ball", "dog", "cat", "hat", "doll", "box", "kite", "cup", "car",
         "fish", "frog", "cake", "key", "drum", "bell", "duck", "bear",
         "boat", "book", "shoe"}

PROBES = [
    ("Lily had a red ball. She threw the",           {"ball", "red"},  False),
    ("Sara had a blue hat. Later, Sara wore the",     {"hat", "blue"},  False),
    ("Ben found a small key. Ben used the",           {"key", "small"}, False),
    ("Lily had a red ball. she went to the park. Then Lily threw the red",
     NOUNS, True),
]

torch.manual_seed(42)


def load(ckpt_path, use_qr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = {k: (v.float() if v.is_floating_point() else v)
              for k, v in ckpt["model"].items()}
    cfg = get_config()
    cfg.model.dim = 64; cfg.model.hidden_dim = 256; cfg.model.max_seq_len = SEQ
    cfg.model.use_query_read = use_qr
    cfg.memory.slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    model = SpiderWeb(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    return model, device, ctx, ckpt


def topk(model, sp, device, ctx, prompt, k=5):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        out = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(sp.DecodeIds([i.item()]).strip(), v.item()) for i, v in zip(idxs, vals)], out


def hit(tokens, wanted):
    for tok, _ in tokens:
        t = tok.lower()
        for w in wanted:
            if w and w in t:
                return tok
    return None


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    arms = {}
    for name, path, uq in ARMS:
        try:
            arms[name] = load(path, uq)
        except FileNotFoundError:
            print(f"!! {name}: checkpoint missing ({path})"); return

    for name, path, _ in ARMS:
        ckpt = arms[name][3]
        ema = ckpt.get("ema_ce"); ema = f"{ema:.3f}" if isinstance(ema, float) else str(ema)
        print(f"{name}: {path}  step={ckpt.get('step')}  ema_ce={ema}")

    summary = []
    for prompt, wanted, is_comp in PROBES:
        print(f"\n{'='*72}")
        print(("COMPETENCE CHECK (noun after 'the red ___')" if is_comp else "BINDING PROBE"))
        print(f'Prompt: "{prompt}"')
        print(f"{'-'*72}")
        row = {"prompt": prompt, "is_comp": is_comp}
        for name, _, _ in ARMS:
            model, device, ctx, _ = arms[name]
            t5, _ = topk(model, sp, device, ctx, prompt)
            h = hit(t5, wanted)
            row[name] = h
            toks = "  ".join(f"{tok!r}:{p:.3f}" for tok, p in t5)
            print(f"  [{name}] {toks}")
            print(f"          => {'HIT '+repr(h) if h else 'absent'}")
        summary.append(row)

    # ── decisive comparison ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("DECISIVE COMPARISON — specific earlier entity in top-5?")
    print(f"{'='*72}")
    print(f"{'Probe':<54}{'A:ON':>9}{'B:OFF':>9}")
    print(f"{'-'*72}")
    binding_rows = [r for r in summary if not r["is_comp"]]
    a_only = both = 0
    for r in binding_rows:
        a, b = r["A:ON "], r["B:OFF"]
        if a and not b: a_only += 1
        if a and b: both += 1
        short = (r["prompt"][:51] + "...") if len(r["prompt"]) > 54 else r["prompt"]
        print(f"{short:<54}{(repr(a) if a else 'absent'):>9}{(repr(b) if b else 'absent'):>9}")

    comp = [r for r in summary if r["is_comp"]][0]
    print(f"\nCompetence (both should hit a noun): "
          f"A:ON={'OK' if comp['A:ON '] else 'FAIL'}  B:OFF={'OK' if comp['B:OFF'] else 'FAIL'}")

    # ── learned ring bias (orbital depth-weighting) ───────────────────────────
    print(f"\n{'='*72}")
    print("LEARNED PER-RING BIAS (Arm A) — did the orbital read favour inner rings?")
    print(f"{'='*72}")
    ck = arms["A:ON "][3]
    bias = ck.get("ring_read_bias")
    mass = ck.get("ring_read_mass")
    if bias is None:
        m = arms["A:ON "][0]
        bias = m.query_read.ring_bias.detach().cpu().tolist()
    NR = len(bias)
    for r, b in enumerate(bias):
        tag = "[inner]" if r < NR // 2 else "[outer]"
        print(f"  Ring {r} {tag}: bias={b:+.4f}"
              + (f"  realized_mass={mass[r]:.3f}" if mass else ""))
    inner = sum(bias[:NR // 2]); outer = sum(bias[NR // 2:])
    print(f"  inner-sum={inner:+.4f}  outer-sum={outer:+.4f}  -> "
          + ("FAVORS INNER (orbital depth-weighting emerged)" if inner > outer + 1e-3
             else "FAVORS OUTER" if outer > inner + 1e-3 else "≈ uniform (no depth preference learned)"))

    print(f"\n{'-'*72}")
    if a_only > 0:
        print(f"✓ VERDICT: orbital query read (A) recalled the entity on {a_only}/{len(binding_rows)} "
              f"probe(s) where the control (B) did not.")
        print("  -> content-addressable read produces real cross-sentence binding.")
    elif both > 0 and both == len(binding_rows):
        print("~ VERDICT: both arms recall equally -> binding came from data/fine-tune, not the read.")
    else:
        print("✗ VERDICT: neither arm recalls the specific entity in top-5.")
        print("  -> content-addressable read alone does not solve binding at this scale/budget.")


if __name__ == "__main__":
    main()
