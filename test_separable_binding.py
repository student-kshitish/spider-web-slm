"""
test_separable_binding.py — DECISIVE binding probe: separable vs blend.

Loads the two finished arms:
  A:SEP   checkpoints/separable_run/separable/best.pt   (write_mode=separable)
  B:BLEND checkpoints/separable_run/blend/best.pt       (write_mode=blend, control)

and runs identical cross-sentence binding probes side by side. For the separable
arm it ALSO reports whether the learned write-gate actually fires on the entity
tokens (ball/hat/key/car) — i.e. did the mechanism store the entities, or
something else?

Inference only. Run:  python3 test_separable_binding.py
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
    ("A:SEP",   "checkpoints/separable_run/separable/best.pt", "separable"),
    ("B:BLEND", "checkpoints/separable_run/blend/best.pt",     "blend"),
]

# (prompt, wanted-tokens, entity-noun-to-track, is_competence)
PROBES = [
    ("Lily had a red ball. She threw the",       {"ball", "red"},  "ball", False),
    ("Sara had a blue hat. Later, Sara wore the", {"hat", "blue"},  "hat",  False),
    ("Ben found a small key. Ben used the",       {"key"},          "key",  False),
    ("Tom had a green car. Tom drove the",        {"car"},          "car",  False),
    ("Lily had a red ball. she went to the park. Then Lily threw the red",
     {"ball","dog","cat","hat","box","car","key","cake","kite","cup"}, None, True),
]

torch.manual_seed(42)


def load(ckpt_path, write_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = {k: (v.float() if v.is_floating_point() else v)
              for k, v in ckpt["model"].items()}
    cfg = get_config()
    cfg.model.dim = 64; cfg.model.hidden_dim = 256; cfg.model.max_seq_len = SEQ
    cfg.model.use_struct_read = False
    cfg.model.use_query_read  = False
    cfg.memory.slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg.memory.write_mode = write_mode
    model = SpiderWeb(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    return model, device, ctx, ckpt


def topk(model, sp, device, ctx, prompt, gate_capture=None, k=5):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    # hook to capture per-position write-gate (separable arm only)
    handle = None
    cap = {}
    if gate_capture is not None:
        sm = model.separable_mem
        def hook(_mod, _inp, _out):
            x = _inp[0]                                  # (B,T,d) input to separable_mem
            g = torch.sigmoid(sm.gate_mlp(x)).squeeze(-1)[0]  # (T,)
            cap["gate"] = g.detach().float().cpu()
        handle = sm.register_forward_hook(hook)

    with torch.no_grad(), ctx:
        out = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    if handle is not None:
        handle.remove()
        gate_capture["gate"] = cap.get("gate")
        gate_capture["ids"]  = ids

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


def gate_on_entity(sp, gate, ids, entity):
    """Find the token position(s) for `entity` and report gate there vs the rest."""
    if gate is None:
        return None
    pieces = [sp.DecodeIds([i]).strip().lower() for i in ids]
    ent_pos = [i for i, p in enumerate(pieces) if entity in p and p != ""]
    if not ent_pos:
        # entity may be split across subwords; fall back to substring over join
        return None
    g = gate
    ent_gate = max(g[i].item() for i in ent_pos)
    seq_mean = g.mean().item()
    # rank of the entity position among all positions (1 = highest gate)
    order = sorted(range(len(g)), key=lambda i: -g[i].item())
    best_ent = max(ent_pos, key=lambda i: g[i].item())
    rank = order.index(best_ent) + 1
    # the top-gated token in the sequence
    top_tok = pieces[order[0]] or sp.DecodeIds([ids[order[0]]]).strip()
    return dict(ent_gate=ent_gate, seq_mean=seq_mean, rank=rank, n=len(g),
                top_tok=top_tok, top_gate=g[order[0]].item(),
                ent_tok=pieces[best_ent])


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    arms = {}
    for name, path, wm in ARMS:
        try:
            arms[name] = (load(path, wm), wm)
        except FileNotFoundError:
            print(f"!! {name}: checkpoint missing ({path})"); return

    print("Loaded:")
    for name, path, _ in ARMS:
        (m, d, c, ck), wm = arms[name]
        ema = ck.get("ema_ce"); ema = f"{ema:.3f}" if isinstance(ema, float) else str(ema)
        print(f"  {name:8} step={ck.get('step')}  ema_ce={ema}  write_mode={wm}  ({path})")

    summary = []
    gate_rows = []
    for prompt, wanted, entity, is_comp in PROBES:
        print(f"\n{'='*74}")
        print("COMPETENCE CHECK" if is_comp else "BINDING PROBE")
        print(f'Prompt: "{prompt}"')
        print(f"{'-'*74}")
        row = {"prompt": prompt, "is_comp": is_comp, "entity": entity}
        for name, _, _ in ARMS:
            (model, device, ctx, _), wm = arms[name]
            gc = {} if wm == "separable" else None
            t5 = topk(model, sp, device, ctx, prompt, gate_capture=gc)
            h = hit(t5, wanted)
            row[name] = h
            toks = "  ".join(f"{tok!r}:{p:.3f}" for tok, p in t5)
            print(f"  [{name:8}] {toks}")
            print(f"            => {'HIT '+repr(h) if h else 'absent'}")
            if gc is not None and entity is not None:
                info = gate_on_entity(sp, gc.get("gate"), gc.get("ids"), entity)
                if info:
                    gate_rows.append((entity, info))
                    fired = "FIRES ✓" if info["ent_gate"] > 0.5 else (
                            "elevated" if info["ent_gate"] > 2*info["seq_mean"] else "low ✗")
                    print(f"            gate@'{info['ent_tok']}'={info['ent_gate']:.3f}  "
                          f"(seq-mean {info['seq_mean']:.3f}, rank {info['rank']}/{info['n']})  "
                          f"-> {fired}   [top-gate token: '{info['top_tok']}' {info['top_gate']:.3f}]")
        summary.append(row)

    # ── DECISIVE COMPARISON ──────────────────────────────────────────────────
    print(f"\n{'='*74}\nDECISIVE COMPARISON — specific entity in top-5?\n{'='*74}")
    print(f"{'Probe (entity)':<40}{'A:SEP':>10}{'B:BLEND':>10}")
    print(f"{'-'*74}")
    binders = [r for r in summary if not r["is_comp"]]
    a_hits = sum(1 for r in binders if r["A:SEP"])
    b_hits = sum(1 for r in binders if r["B:BLEND"])
    for r in binders:
        a, b = r["A:SEP"], r["B:BLEND"]
        lab = f"{r['prompt'][:28]}... ({r['entity']})"
        print(f"{lab:<40}{(repr(a) if a else 'absent'):>10}{(repr(b) if b else 'absent'):>10}")
    comp = [r for r in summary if r["is_comp"]][0]
    print(f"\nCompetence: A:SEP={'OK' if comp['A:SEP'] else 'FAIL'}  "
          f"B:BLEND={'OK' if comp['B:BLEND'] else 'FAIL'}")
    print(f"Binding hits:  A:SEP={a_hits}/{len(binders)}   B:BLEND={b_hits}/{len(binders)}")

    # ── GATE-FIRING SUMMARY ──────────────────────────────────────────────────
    print(f"\n{'-'*74}\nGATE-FIRING (separable arm): did the gate store the entity tokens?")
    if gate_rows:
        fired = sum(1 for _, i in gate_rows if i["ent_gate"] > 0.5)
        elev  = sum(1 for _, i in gate_rows if 0.5 >= i["ent_gate"] > 2*i["seq_mean"])
        for ent, i in gate_rows:
            tag = "FIRES ✓" if i["ent_gate"] > 0.5 else (
                  "elevated" if i["ent_gate"] > 2*i["seq_mean"] else "low ✗")
            print(f"  {ent:<5} gate={i['ent_gate']:.3f} (vs seq-mean {i['seq_mean']:.3f}, "
                  f"rank {i['rank']}/{i['n']})  {tag}")
        print(f"  => gate fires (>0.5) on {fired}/{len(gate_rows)} entity tokens; "
              f"elevated (>2x mean) on {elev} more.")
    else:
        print("  (no gate captured)")

    # ── VERDICT ──────────────────────────────────────────────────────────────
    print(f"\n{'#'*74}\nVERDICT\n{'#'*74}")
    print(f"  Q1 separable retrieves entity where blend doesn't:  "
          f"A:SEP={a_hits}/4  vs  B:BLEND={b_hits}/4")
    if a_hits == 0 and b_hits == 0:
        print("  => Neither arm binds. The write-side redesign did NOT crack cross-")
        print("     sentence binding at this scale/budget. (CE edge did not translate")
        print("     into entity retrieval.)")
    elif a_hits > b_hits:
        consistent = "CONSISTENT (multiple probes)" if a_hits >= 3 else \
                     ("partial" if a_hits == 2 else "SINGLE probe — likely NOISE")
        print(f"  => Separable binds MORE than blend ({a_hits} vs {b_hits}) — {consistent}.")
    elif a_hits == b_hits and a_hits > 0:
        print(f"  => Both bind equally ({a_hits}); separable shows no binding advantage.")
    else:
        print(f"  => Blend binds more ({b_hits} vs {a_hits}) — redesign did not help.")


if __name__ == "__main__":
    main()
