"""
GENERALIZATION GRADIENT — EVALUATION ONLY (no training, nothing modified).
The off-template eval showed the finished chain works on templates and fails on
T1 (new verbs/frames). This measures WHERE between the two the learned components
break, by walking a graded ladder of increasing lexical/structural novelty and
watching the gate (lambda) and the locator (locObj) degrade.

Structure (k=2..3 name-object intros + 1-2 fillers + name-cued recall) and filler
handling are held CONSTANT across every rung, so the ONLY thing that changes rung
to rung is the novelty axis added — the ladder is cumulative:

  G0  exact training templates            (control — the 74.2%-regime)
  G1  + ONE novel adjective per item      (training frames/verbs otherwise)
  G2  + ALL adjectives novel              (training frames/verbs)
  G3  + novel recall verbs                (training frames)
  G4  + novel intro & recall FRAMES       (= the old T1)
  G5  + reordered intro sentences         (T1.5; harder than T1, easier than T2 clefts)

Per rung: mixture top-5, 2AFC, lambda[recall], %gen-dominant (lambda>0.5), readCue,
locObj.  Held-out nouns throughout.  Answers: CLIFF (memorized distribution — fine
until a threshold then collapse) or GRADIENT (partial abstraction — smooth decay)?
And at which novelty axis does lambda start rising / locObj start falling?

Sanity: loader from meta, name_src=None asserted, deterministic per-rung seeds,
2 sample items printed per rung, no checkpoint modified.

Run:  python3 probe_generalization_gradient.py
"""
import os, sys, json, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm

import probe_binding_linear as P      # loader (reused)
import probe_multi_eval as M          # tok_pos / name_pos_before / single_tok (reused)
import train_attn_super as TA

SEED = 0
N    = 80
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/content_addr_A/best.pt"

FEMALE, MALE = TA.FEMALE, TA.MALE
ATTRS_TR = TA.ATTRS
VERBS_TR = TA.VERBS
FRAMES_TR = ["{name} had a {attr} {obj}.", "{name} found a {attr} {obj}.",
             "{name} got a {attr} {obj}.", "One day {name} saw a {attr} {obj}."]
RECALL_TR = ["Then {name} {verb} the", "At last {name} {verb} the",
             "Later {name} {verb} the"]                    # trained skeletons, end at "the"

ATTRS_NOV = ["purple", "orange", "pink", "huge", "dark", "smooth", "heavy",
             "cold", "striped", "spotted", "wooden", "silver", "golden", "furry"]
VERBS_NOV = ["tossed", "lifted", "snatched", "fetched", "waved", "chose", "nudged", "spun"]
FRAMES_NOV = ["{name} was holding a {attr} {obj}.", "{name} carried a {attr} {obj}.",
              "{name} brought a {attr} {obj}.", "{name} was playing with a {attr} {obj}.",
              "{name} showed off a {attr} {obj}.", "{name} bought a {attr} {obj}."]
RECALL_NOV = ["Soon {name} {verb} the", "Finally {name} {verb} the",
              "Quickly {name} {verb} the", "In the end {name} {verb} the"]

# cumulative ladder configs: each rung adds exactly one novelty axis.
RUNGS = [
    ("G0 template",  dict(frames=FRAMES_TR,  attrs=ATTRS_TR,  one_novel=False,
                          verbs=VERBS_TR,  recall=RECALL_TR,  reorder=False)),
    ("G1 +1 adj",    dict(frames=FRAMES_TR,  attrs=ATTRS_TR,  one_novel=True,
                          verbs=VERBS_TR,  recall=RECALL_TR,  reorder=False)),
    ("G2 all-adj",   dict(frames=FRAMES_TR,  attrs=ATTRS_NOV, one_novel=False,
                          verbs=VERBS_TR,  recall=RECALL_TR,  reorder=False)),
    ("G3 +verbs",    dict(frames=FRAMES_TR,  attrs=ATTRS_NOV, one_novel=False,
                          verbs=VERBS_NOV, recall=RECALL_TR,  reorder=False)),
    ("G4 +frames",   dict(frames=FRAMES_NOV, attrs=ATTRS_NOV, one_novel=False,
                          verbs=VERBS_NOV, recall=RECALL_NOV, reorder=False)),
    ("G5 +reorder",  dict(frames=FRAMES_NOV, attrs=ATTRS_NOV, one_novel=False,
                          verbs=VERBS_NOV, recall=RECALL_NOV, reorder=True)),
]


def gen_rung(rng, objs, C):
    k = rng.choice([2, 3])
    names = rng.sample(FEMALE + MALE, k)
    chosen = rng.sample(objs, k)
    if C["one_novel"]:                                    # G1: exactly one novel adjective
        attrs = [rng.choice(ATTRS_TR) for _ in range(k)]
        attrs[rng.randrange(k)] = rng.choice(ATTRS_NOV)
    else:
        attrs = [rng.choice(C["attrs"]) for _ in range(k)]
    intros = [(chosen[i], names[i],
               rng.choice(C["frames"]).format(name=names[i], attr=attrs[i], obj=chosen[i]))
              for i in range(k)]
    if C["reorder"]:
        rng.shuffle(intros)                               # G5: vary intro sentence order
    parts = [t for _, _, t in intros]
    p0 = intros[0][1]                                     # constant filler treatment (as training)
    pr = ("she", "She") if p0 in FEMALE else ("he", "He")
    for _ in range(rng.randint(1, 2)):
        parts.append(rng.choice(TA.FILLER).format(pron=pr[0], Pron=pr[1]))
    cue = rng.randrange(len(intros) - 1)                  # NON-last appearance -> recency is wrong
    rec = len(intros) - 1
    cname = intros[cue][1]
    parts.append(rng.choice(C["recall"]).format(name=cname, verb=rng.choice(C["verbs"])))
    ent_pairs = [(o, nm) for o, nm, _ in intros]
    return (" ".join(parts), intros[cue][0], intros[rec][0], ent_pairs, cname)


@torch.no_grad()
def measure(model, device, sp, items):
    cue5 = afc = n = 0
    lam_sum = gen_dom = read_sum = 0.0
    lo_hit = lo_tot = 0
    for text, cobj, robj, ent_pairs, cue_name in items:
        pieces = sp.encode(text, out_type=str)
        ids = sp.EncodeAsIds(text)
        cs = M.tok_pos(pieces, cobj)
        cid = sp.piece_to_id("▁" + cobj); rid = sp.piece_to_id("▁" + robj)
        if cs is None or cid == sp.unk_id() or rid == sp.unk_id():
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        name_src = None
        assert name_src is None, "name_src must be None at inference"
        out = model(inp, tau=0.1, hard=True, use_pointer=True, subj_id=None, name_src=name_src)
        r = len(ids) - 1
        Pv = out["pointer"]["P"][0, r].float()
        logits = torch.log(Pv.clamp_min(1e-9))
        cue5 += ((logits > logits[cid]).sum().item() + 1) <= 5
        afc  += logits[cid] > logits[rid]
        lam = out["pointer"]["lambda"][0, r].item()
        lam_sum += lam; gen_dom += (lam > 0.5)
        read_sum += out["sep_stats"]["read_dist"][0, r].float()[cs].item()
        ns = out.get("name_stats")
        if ns is not None:
            a = ns["attn"][0].float()
            for ob, nm in ent_pairs:
                op = M.tok_pos(pieces, ob)
                if op is None:
                    continue
                tp = M.name_pos_before(pieces, sp, nm, op)
                if tp is None:
                    continue
                lo_tot += 1
                lo_hit += int(a[op].argmax().item() == tp)
        n += 1
    if n == 0:
        return None
    return dict(n=n, top5=cue5 / n, afc=afc / n, lam=lam_sum / n,
                gendom=gen_dom / n, readCue=read_sum / n,
                locObj=(lo_hit / lo_tot) if lo_tot else float("nan"))


def main():
    torch.manual_seed(SEED); random.seed(SEED)
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    vocab = json.load(open("data/wide_vocab.json"))
    held = M.single_tok(sp, vocab["test"])

    P.CKPT = CKPT
    model, device, ema = P.load_model()
    meta = torch.load(CKPT, map_location="cpu")
    assert bool(meta.get("use_pointer")) and bool(meta.get("mem_copy")), "need pointer+mem_copy"
    assert not bool(meta.get("name_transport")) and not bool(meta.get("oracle_bind"))
    print(f"[grad] ckpt={CKPT} ema_ce={ema:.4f} name_lookback={bool(meta.get('name_lookback'))} "
          f"n={N}/rung held-out nouns={len(held)}  (structure/fillers held CONSTANT; cumulative novelty)")

    # build all rungs (deterministic per-rung seed)
    built = []
    for i, (tag, C) in enumerate(RUNGS):
        rng = random.Random(SEED + 100 + i)
        items = [gen_rung(rng, held, C) for _ in range(N)]
        built.append((tag, items))

    print("\n" + "=" * 96)
    print("SAMPLE ITEMS (2/rung) — cue names a NON-last subject; recency = last object = WRONG answer")
    print("=" * 96)
    for tag, items in built:
        print(f"\n[{tag}]")
        for it in items[:2]:
            print(f"  {it[0]} ___   (answer={it[1]!r}  recency={it[2]!r})")

    rows = [(tag, measure(model, device, sp, items)) for tag, items in built]

    print("\n" + "=" * 96)
    print("GENERALIZATION GRADIENT  (content_addr_A; held-out nouns; cumulative novelty ladder)")
    print("=" * 96)
    print(f"{'rung':<14} {'mixTop5':>8} {'2AFC':>7} {'lam[rec]':>9} {'%gen>.5':>9} "
          f"{'readCue':>8} {'locObj':>8} {'n':>5}")
    print("-" * 96)
    for tag, m in rows:
        if m is None:
            print(f"{tag:<14} (no scorable items)"); continue
        lo = "    nan " if m['locObj'] != m['locObj'] else f"{100*m['locObj']:>6.1f}% "
        print(f"{tag:<14} {100*m['top5']:>7.1f}% {100*m['afc']:>6.1f}% {m['lam']:>9.2f} "
              f"{100*m['gendom']:>8.1f}% {m['readCue']:>8.2f} {lo:>8} {m['n']:>5}")
    print("-" * 96)
    print("mixTop5 = final emit | 2AFC = P(cue>recency), chance 50% | lam[rec] = gate on GEN branch "
          "(high = copy suppressed)\n%gen>.5 = frac gate generative-dominant | readCue = read mass on "
          "source | locObj = name-localization accuracy\nCLIFF = flat then sudden drop at one axis | "
          "GRADIENT = smooth decay; watch WHERE lam rises and locObj falls.")


if __name__ == "__main__":
    main()
