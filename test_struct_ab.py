"""
Step 3 test — side-by-side A/B of the structural (direction-gated) read.

  Arm A (struct mask ON)  : checkpoints/struct_on/best.pt
  Arm B (flat, mask OFF)  : checkpoints/struct_off/best.pt

Decisive questions:
  1. Does EITHER arm now bind (retrieve the specific earlier entity)? i.e. did
     position-resolved memory finally enable binding at all?
  2. Does the structural mask (A) help / hurt / match flat retrieval (B)?
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
    ("A:MASK", "checkpoints/struct_on/best.pt",  True),
    ("B:FLAT", "checkpoints/struct_off/best.pt", False),
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


def load(ckpt_path, struct_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = {k: (v.float() if v.is_floating_point() else v)
              for k, v in ckpt["model"].items()}
    cfg = get_config()
    cfg.model.dim = 64; cfg.model.hidden_dim = 256; cfg.model.max_seq_len = SEQ
    cfg.model.use_struct_read = True
    cfg.model.struct_mask = struct_mask
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
    arms = {}
    for name, path, sm in ARMS:
        try:
            arms[name] = load(path, sm)
        except FileNotFoundError:
            print(f"!! {name}: checkpoint missing ({path})"); return

    for name, path, _ in ARMS:
        ck = arms[name][3]
        ema = ck.get("ema_ce"); ema = f"{ema:.3f}" if isinstance(ema, float) else str(ema)
        print(f"{name}: {path}  step={ck.get('step')}  ema_ce={ema}  "
              f"struct_bias={ck.get('struct_read_bias')}")

    summary = []
    for prompt, wanted, is_comp in PROBES:
        print(f"\n{'='*72}")
        print(("COMPETENCE CHECK (noun after 'the red ___')" if is_comp else "BINDING PROBE"))
        print(f'Prompt: "{prompt}"')
        print(f"{'-'*72}")
        row = {"prompt": prompt, "is_comp": is_comp}
        for name, _, _ in ARMS:
            model, device, ctx, _ = arms[name]
            t5 = topk(model, sp, device, ctx, prompt)
            h = hit(t5, wanted)
            row[name] = h
            toks = "  ".join(f"{tok!r}:{p:.3f}" for tok, p in t5)
            print(f"  [{name}] {toks}")
            print(f"          => {'HIT '+repr(h) if h else 'absent'}")
        summary.append(row)

    print(f"\n{'='*72}")
    print("DECISIVE COMPARISON — specific earlier entity in top-5?")
    print(f"{'='*72}")
    print(f"{'Probe':<54}{'A:MASK':>9}{'B:FLAT':>9}")
    print(f"{'-'*72}")
    binding_rows = [r for r in summary if not r["is_comp"]]
    a_hits = sum(1 for r in binding_rows if r["A:MASK"])
    b_hits = sum(1 for r in binding_rows if r["B:FLAT"])
    for r in binding_rows:
        a, b = r["A:MASK"], r["B:FLAT"]
        short = (r["prompt"][:51] + "...") if len(r["prompt"]) > 54 else r["prompt"]
        print(f"{short:<54}{(repr(a) if a else 'absent'):>9}{(repr(b) if b else 'absent'):>9}")
    comp = [r for r in summary if r["is_comp"]][0]
    print(f"\nCompetence (both should hit a noun): "
          f"A:MASK={'OK' if comp['A:MASK'] else 'FAIL'}  B:FLAT={'OK' if comp['B:FLAT'] else 'FAIL'}")

    print(f"\n{'-'*72}")
    print(f"Binding hits:  A:MASK={a_hits}/{len(binding_rows)}   B:FLAT={b_hits}/{len(binding_rows)}")
    if a_hits == 0 and b_hits == 0:
        print("✗ Q1: neither arm binds — position-resolved retrieval alone did NOT enable")
        print("      binding at this scale/budget.")
    else:
        print(f"✓ Q1: position-resolved retrieval ENABLED binding (A={a_hits}, B={b_hits} hits).")
        if a_hits > b_hits:
            print("  Q2: structural mask HELPS (A > B).")
        elif b_hits > a_hits:
            print("  Q2: structural mask HURTS (flat B > masked A).")
        else:
            print("  Q2: structural mask MATCHES flat (A == B).")


if __name__ == "__main__":
    main()
