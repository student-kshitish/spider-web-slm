"""
Step 3 — Cross-sentence binding probes across the hybrid A/B arms.

For each arm we ask the SAME question that motivated the whole project: after
"<Name> had a <adj> <object>.  <distractor sentence>.  <Name> <verb> the", does
the bound entity (ball / hat / key / car, and its adjective) appear in the top-5
next tokens?  Prompts mirror the binding.txt training distribution exactly.

Arms (each loaded with its own hybrid config so the forward matches training):
  baseline : checkpoints/final_run/best.pt        use_hybrid=False
  off      : checkpoints/hybrid_off/best.pt        use_hybrid=False  (control re-trained)
  bounded  : checkpoints/hybrid_bounded/best.pt    use_hybrid=True, width=32
  full     : checkpoints/hybrid_full/best.pt       use_hybrid=True, width=full

Also prints, for the ON arms, the gate FIRING RATE on the probe prompts — the
efficiency readout (how often attention actually fired vs full attention).

Run:  python3 test_hybrid_binding.py
"""

import torch
import torch.nn.functional as F
import sentencepiece as spm
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb

SEQ = 256
TAU = 0.1

# (checkpoint, use_hybrid, lookback_width)
ARMS = [
    ("baseline", "checkpoints/final_run/best.pt",     False, 32),
    ("off",      "checkpoints/hybrid_off/best.pt",     False, 32),
    ("bounded",  "checkpoints/hybrid_bounded/best.pt", True,  32),
    ("full",     "checkpoints/hybrid_full/best.pt",    True,  -1),
]

# Prompts in the exact binding.txt shape: setup . distractor . recall-cue ___
PROMPTS = [
    ("Lily had a red ball. Birds flew in the sky. Lily picked up the",
     {"ball", "red"}),
    ("Ben found a round hat. The sun was warm and bright. Ben loved the",
     {"hat", "round"}),
    ("Tom got a shiny key. she ran down the hill. Tom found the",
     {"key", "shiny"}),
    ("Zoe had a blue car. Birds flew in the sky. Zoe wanted the",
     {"car", "blue"}),
]

torch.manual_seed(42)


def load(ckpt_path, use_hybrid, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = {k: v.float() if v.is_floating_point() else v
              for k, v in ckpt["model"].items()}

    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = SEQ
    cfg.model.use_hybrid     = use_hybrid
    cfg.model.lookback_width = width
    seed_key = "rings.0.0.memory.m_t_seed"
    if seed_key in state:
        cfg.memory.slots = state[seed_key].shape[0]
    model = SpiderWeb(cfg).to(device)
    # strict=False: baseline predates the new modules; they are identity at init
    # and (for use_hybrid=False arms) never touch the logits path.
    missing, _ = model.load_state_dict(state, strict=False)
    new_pref = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj")
    bad = [k for k in missing if not k.startswith(new_pref)]
    if bad:
        print(f"  [warn] unexpected missing keys: {bad}")
    model.eval()
    ctx = (torch.autocast("cuda", torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    return model, device, ctx, ckpt.get("step", "?"), ckpt.get("ema_ce", None)


def topk_and_fire(model, sp, device, ctx, prompt, use_hybrid, k=5):
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), ctx:
        out    = model(inp[:, -SEQ:], tau=TAU, hard=True)
        logits = out["logits"][0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    t5 = [(sp.DecodeIds([i.item()]).strip(), v.item()) for i, v in zip(idxs, vals)]
    fire = None
    if use_hybrid and out.get("hybrid_stats"):
        fire = out["hybrid_stats"]["gate_frac_on"]   # frac of prompt tokens flagged
    return t5, fire


def hit(tokens, wanted):
    for tok, _ in tokens:
        t = tok.lower()
        for w in wanted:
            if w in t:
                return tok
    return None


def run_arm(name, ckpt_path, use_hybrid, width, sp):
    try:
        model, device, ctx, step, ema = load(ckpt_path, use_hybrid, width)
    except FileNotFoundError:
        print(f"\n### {name}: checkpoint not found ({ckpt_path}) — skipping")
        return None
    ema_s = f"{ema:.3f}" if isinstance(ema, float) else str(ema)
    wstr = "full" if width <= 0 else str(width)
    print(f"\n{'='*68}")
    print(f"### {name}   ({ckpt_path})   step={step} ema_ce={ema_s} "
          f"hybrid={use_hybrid} width={wstr}")
    print(f"{'='*68}")
    results, fires = {}, []
    for prompt, wanted in PROMPTS:
        t5, fire = topk_and_fire(model, sp, device, ctx, prompt, use_hybrid)
        h = hit(t5, wanted)
        results[prompt] = h
        if fire is not None:
            fires.append(fire)
        fstr = f"   [gate fire {100*fire:.0f}%]" if fire is not None else ""
        print(f'\nPrompt: "{prompt}"{fstr}')
        print(f"  want one of: {sorted(wanted)}")
        for rank, (tok, prob) in enumerate(t5, 1):
            mark = "  <-- HIT" if (h is not None and tok == h) else ""
            print(f"    {rank}. {tok!r:<16} {prob:.4f}{mark}")
        print(f"  => {'RECALLED: ' + repr(h) if h else 'ABSENT (not in top-5)'}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean_fire = (sum(fires) / len(fires)) if fires else None
    return {"results": results, "mean_fire": mean_fire}


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")

    arm_out = {}
    for name, ckpt, uh, w in ARMS:
        arm_out[name] = run_arm(name, ckpt, uh, w, sp)

    present = [n for n in arm_out if arm_out[n] is not None]
    if not present:
        print("\nNo checkpoints found — run train_hybrid_ab.py first.")
        return

    print(f"\n{'='*68}")
    print("SIDE-BY-SIDE — bound entity in top-5?")
    print(f"{'='*68}")
    head = f"{'Prompt':<40}" + "".join(f"{n:>10}" for n in present)
    print(head); print("-" * len(head))
    totals = {n: 0 for n in present}
    for prompt, _ in PROMPTS:
        short = (prompt[:37] + "...") if len(prompt) > 40 else prompt
        row = f"{short:<40}"
        for n in present:
            h = arm_out[n]["results"][prompt]
            row += f"{(repr(h) if h else 'absent'):>10}"
            if h:
                totals[n] += 1
        print(row)
    print("-" * len(head))
    print(f"{'TOTAL hits / ' + str(len(PROMPTS)):<40}" +
          "".join(f"{str(totals[n]):>10}" for n in present))

    # efficiency note
    print(f"\n{'-'*68}")
    print("EFFICIENCY — mean gate firing rate on probe prompts (ON arms only):")
    for n in present:
        mf = arm_out[n]["mean_fire"]
        if mf is not None:
            print(f"  {n:<10}: {100*mf:5.1f}%  of tokens invoked attention "
                  f"({'surgical' if mf < 0.5 else 'near-dense'})")
    print(f"{'-'*68}")


if __name__ == "__main__":
    main()
