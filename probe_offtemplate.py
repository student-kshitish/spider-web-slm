"""
OFF-TEMPLATE binding eval — how far does the finished binding chain generalize
OFF the training templates?  EVALUATION ONLY: no training, no new modules, no
tuning, no checkpoint is modified.  We reuse probe_multi_eval's loader
(probe_binding_linear.load_model) and its exact scoring math (pointer logits,
mem_copy read-mass, name_lookback name-localization), swapping only the item
GENERATOR, over four tiers of increasing distance from the training distribution.
Held-out (unseen) entity nouns are used THROUGHOUT.

  Tier 1  Surface variation, same structure. New verbs / adjectives / intro &
          recall frames that never appear in the training templates. Same
          underlying structure (name-object intros, name-cued recall).
  Tier 2  Structural noise. 1-3 filler sentences (no entities) between intros
          and recall; shuffled intro order; inverted / cleft clause shapes
          ("A ball was what Lily had"). Stresses the locator's position/structure
          assumptions.
  Tier 3  Pronoun cue (hard probe). Recall cued by a pronoun, disambiguated by
          gender, instead of a name repeat ("Lily had a ball. Tom had a car. She
          threw the ___"). The model was NEVER trained to resolve pronouns —
          expect near-chance; we measure it to document the boundary honestly.
  Tier 4  Natural text recall. Real TinyStories passages where a held-out noun is
          introduced and re-mentioned; the re-mention is masked and we test recall
          of that noun. No template at all — the closest to "real".

For each tier we report the SAME metrics as the standard probe:
  cue-top1/top5 (target-top5 for Tier 4), 2AFC vs distractor, readCue/readRec,
  locObj/locRec (where a name-binding structure exists), plus the template
  baseline row (the standard multi-eval, held-out mixed-k; ~74.2% 2AFC) for
  reference.  3 sample items per tier are printed for fairness inspection.

Sanity: same loader (arch flags restored from checkpoint meta), name_src asserted
None at every forward (the locator must find names unaided), deterministic seeds,
no checkpoint touched.

Run:  python3 probe_offtemplate.py
"""
import os, sys, json, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm

import probe_binding_linear as P      # loader + arch-flag restore (reused)
import probe_multi_eval as M          # tok_pos / name_pos_before / measure (reused)
import train_attn_super as TA         # name pools (FEMALE/MALE)

SEED = 0
N    = 100                            # ~100 examples per tier

FEMALE, MALE = TA.FEMALE, TA.MALE

# ── OFF-TEMPLATE lexicons: none of these frames/verbs/adjectives appear in the
#    training templates (train_attn_super INTRO/RECALL/VERBS/ATTRS/MULTI_RECALL). ──
T1_INTRO = ["{name} was holding a {attr} {obj}.",
            "{name} carried a {attr} {obj}.",
            "{name} brought a {attr} {obj}.",
            "{name} was playing with a {attr} {obj}.",
            "{name} showed off a {attr} {obj}.",
            "{name} bought a {attr} {obj}."]
T1_ATTR  = ["purple", "orange", "pink", "huge", "dark", "smooth", "heavy",
            "cold", "striped", "spotted", "wooden", "silver", "golden", "furry"]
T1_VERB  = ["tossed", "lifted", "snatched", "fetched", "waved", "chose",
            "nudged", "spun"]
T1_RECALL = ["Soon {name} {verb} the", "Finally {name} {verb} the",
             "Quickly {name} {verb} the", "In the end {name} {verb} the"]

# Tier 2: inverted / cleft intro shapes + name-order shuffle + entity-free fillers.
T2_INTRO = ["{name} had a {attr} {obj}.",
            "A {attr} {obj} was what {name} had.",
            "It was {name} who had a {attr} {obj}.",
            "There was a {attr} {obj} that {name} owned.",
            "What {name} had was a {attr} {obj}."]
T2_FILLER = ["The wind blew softly.", "The sun rose over the hills.",
             "Everything was quiet.", "Time passed slowly.",
             "The morning felt calm.", "A gentle breeze moved by.",
             "Nothing much happened for a while.", "The sky turned soft and grey."]
T2_VERB   = TA.VERBS                       # in-distribution surface; STRUCTURE is the stressor
T2_ATTR   = TA.ATTRS

# ── generators: every one returns (text_or_ids, cobj, robj, ent_pairs, cue_name)
#    exactly like probe_multi_eval.make_multi, so measure_gen scores them the
#    same way. text ends right before the object (last token ~ "the"); cobj is
#    the CUE subject's object; robj is the recency (last-introduced) distractor. ──

def gen_tier1(rng, objs):
    k = rng.choice([2, 3])
    names = rng.sample(FEMALE + MALE, k)
    chosen = rng.sample(objs, k)
    parts = [rng.choice(T1_INTRO).format(
        name=names[i], attr=rng.choice(T1_ATTR), obj=chosen[i]) for i in range(k)]
    cue = rng.randrange(k - 1)                       # NON-last -> recency is a distractor
    parts.append(rng.choice(T1_RECALL).format(name=names[cue], verb=rng.choice(T1_VERB)))
    return (" ".join(parts), chosen[cue], chosen[k - 1],
            list(zip(chosen, names)), names[cue])


def gen_tier2(rng, objs):
    k = rng.choice([2, 3])
    names = rng.sample(FEMALE + MALE, k)
    chosen = rng.sample(objs, k)
    intros = [(chosen[i], names[i],
               rng.choice(T2_INTRO).format(name=names[i], attr=rng.choice(T2_ATTR),
                                           obj=chosen[i])) for i in range(k)]
    rng.shuffle(intros)                              # vary intro ORDER
    parts = [c for _, _, c in intros]
    for _ in range(rng.randint(1, 3)):               # 1-3 entity-free fillers
        parts.append(rng.choice(T2_FILLER))
    cue = 0                                           # first-appearing = NON-recent
    rec = k - 1                                       # last-appearing = recency distractor
    cname = intros[cue][1]
    parts.append("Then {name} {verb} the".format(name=cname, verb=rng.choice(T2_VERB)))
    ent_pairs = [(o, nm) for o, nm, _ in intros]
    return (" ".join(parts), intros[cue][0], intros[rec][0], ent_pairs, cname)


def gen_tier3(rng, objs):
    # k=2 with DIFFERENT genders so a pronoun uniquely selects one subject; the
    # cue is the FIRST (non-recent) subject, its pronoun the answer key.
    fem, mal = rng.choice(FEMALE), rng.choice(MALE)
    of, om = rng.sample(objs, 2)
    if rng.random() < 0.5:
        first, firstobj, fpron = fem, of, "She"; second, secondobj = mal, om
    else:
        first, firstobj, fpron = mal, om, "He";  second, secondobj = fem, of
    parts = ["{n} had a {a} {o}.".format(n=first,  a=rng.choice(TA.ATTRS), o=firstobj),
             "{n} had a {a} {o}.".format(n=second, a=rng.choice(TA.ATTRS), o=secondobj)]
    parts.append("{P} {v} the".format(P=fpron, v=rng.choice(TA.VERBS)))
    return (" ".join(parts), firstobj, secondobj,
            [(firstobj, first), (secondobj, second)], first)


# ── Tier 4: real TinyStories. Held-out noun introduced then re-mentioned; mask
#    the re-mention. Returns token IDs (not a re-tokenized string) so the mask
#    boundary is exact. Distractor = the most-recent OTHER single-tok noun before
#    the mask (a recency distractor, so 2AFC isn't won by "emit the recent noun").
DET = {"▁the", "▁a", "▁his", "▁her", "▁my", "▁your", "▁that", "▁this",
       "▁another", "▁one", "▁some", "▁little", "▁big", "▁small", "▁old", "▁new"}
# verb/function-word homographs that also live in the noun vocab — never a fair
# recency distractor (e.g. past-tense "saw", modal "can", "may", "well").
HOMOGRAPH = {"saw", "can", "may", "well", "back", "park", "spring", "fall", "wave"}


def build_tier4(sp, held, allnouns, n_target, seed):
    rng = random.Random(seed)
    text = open("data/raw/tinystories.txt").read()
    # single-story paragraph pieces (split \n\n into paragraphs, then \n so a
    # story boundary can't sit inside a piece -> a repeated rare noun is same-entity)
    paras = [p for blk in text.split("\n\n") for p in blk.split("\n") if p.strip()]
    rng.shuffle(paras)
    held_pieces = {"▁" + w: w for w in held}
    items, samples = [], []
    for p in paras:
        if len(items) >= n_target:
            break
        pcs = sp.encode(p, out_type=str)
        ids = sp.EncodeAsIds(p)                       # same segmentation as pcs
        if len(pcs) != len(ids):
            continue
        # find a held-out noun with >=2 mentions whose 2nd mention sits in a slot
        best = None
        for w, word in held_pieces.items():
            pos = [i for i, t in enumerate(pcs) if t == w]
            if len(pos) < 2:
                continue
            second = pos[1]
            if second == 0 or pcs[second - 1] not in DET:
                continue
            best = (word, pos[0], second)
            break
        if best is None:
            continue
        word, first, second = best
        # recency distractor: most-recent OTHER single-tok noun before the mask
        distr = None
        for i in range(second - 1, -1, -1):
            t = pcs[i]
            if (t.startswith("▁") and t[1:] in allnouns and t[1:] != word
                    and t[1:] not in HOMOGRAPH):
                distr = t[1:]; break
        prefix_ids = ids[:second]                     # last token is the determiner
        items.append((prefix_ids, word, distr, [], None))
        if len(samples) < 3:
            samples.append((sp.decode(prefix_ids), word, distr))
    return items, samples


# ── generalized measure: probe_multi_eval.measure's scoring, generator-agnostic ──
@torch.no_grad()
def measure_gen(model, device, sp, items, use_ptr, mem_copy, name_lookback):
    cue1 = cue5 = rec1 = 0
    n = n_rec = n_read = 0
    m_cue = m_rec = 0.0
    afc_hit = afc_tot = 0
    lo_hit = lo_tot = 0
    lr_hit = lr_tot = 0
    for first, cobj, robj, ent_pairs, cue_name in items:
        if isinstance(first, str):
            pieces = sp.encode(first, out_type=str)
            ids = sp.EncodeAsIds(first)
        else:
            ids = list(first)
            pieces = [sp.id_to_piece(i) for i in ids]
        cid = sp.piece_to_id("▁" + cobj)
        if cid == sp.unk_id():
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        name_src = None
        assert name_src is None, "name_src must be None at inference (locator finds names unaided)"
        out = model(inp, tau=0.1, hard=True, use_pointer=use_ptr,
                    subj_id=None, name_src=name_src)

        # name-localization (name_lookback): does the lookback attention argmax
        # onto the governing name?  labels used ONLY for scoring, never in forward.
        if name_lookback and ent_pairs and out.get("name_stats") is not None:
            a = out["name_stats"]["attn"][0].float()
            for ob, nm in ent_pairs:
                op = M.tok_pos(pieces, ob)
                if op is None:
                    continue
                tp = M.name_pos_before(pieces, sp, nm, op)
                if tp is None:
                    continue
                lo_tot += 1
                lo_hit += int(a[op].argmax().item() == tp)
            if cue_name is not None:
                rp = len(ids) - 1
                cp = M.name_pos_before(pieces, sp, cue_name, rp)
                if cp is not None:
                    lr_tot += 1
                    lr_hit += int(a[rp].argmax().item() == cp)

        if use_ptr:
            Pv = out["pointer"]["P"][0, -1].float()
            logits = torch.log(Pv.clamp_min(1e-9))
        else:
            logits = out["logits"][0, -1].float()

        cue1 += logits.argmax().item() == cid
        cue5 += ((logits > logits[cid]).sum().item() + 1) <= 5
        n += 1

        if robj is not None:
            rid = sp.piece_to_id("▁" + robj)
            if rid != sp.unk_id():
                rec1 += logits.argmax().item() == rid
                afc_hit += int(logits[cid] > logits[rid]); afc_tot += 1
                n_rec += 1
                # retrieval selectivity: read mass at cue source vs recency source
                cs, rs = M.tok_pos(pieces, cobj), M.tok_pos(pieces, robj)
                row = (out["sep_stats"]["read_dist"][0, -1].float() if mem_copy
                       else None)
                if row is not None and cs is not None and rs is not None:
                    m_cue += row[cs].item(); m_rec += row[rs].item(); n_read += 1
    if n == 0:
        return None
    nan = float("nan")
    return dict(
        cue1=cue1 / n, cue5=cue5 / n,
        rec1=(rec1 / n_rec) if n_rec else nan,
        afc=(afc_hit / afc_tot) if afc_tot else nan,
        readCue=(m_cue / n_read) if n_read else nan,
        readRec=(m_rec / n_read) if n_read else nan,
        locObj=(lo_hit / lo_tot) if lo_tot else nan,
        locRec=(lr_hit / lr_tot) if lr_tot else nan,
        n=n)


def gen_items(gen, objs, n_target, seed):
    rng = random.Random(seed)
    items = []
    guard = 0
    while len(items) < n_target and guard < n_target * 50:
        guard += 1
        items.append(gen(rng, objs))
    return items


def sample_str(it):
    first, cobj, robj = it[0], it[1], it[2]
    txt = first if isinstance(first, str) else None
    return txt, cobj, robj


def pct(x):
    return "   nan " if x != x else f"{100*x:>6.1f}%"


def main():
    torch.manual_seed(SEED); random.seed(SEED)
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    vocab = json.load(open("data/wide_vocab.json"))
    M.N = N                                                # template-base row uses n=N too
    held = M.single_tok(sp, vocab["test"])                 # held-out nouns (used throughout)
    allnouns = set(M.single_tok(sp, TA.OBJECTS + vocab["train"] + vocab["test"]))
    print(f"[offtmpl] held-out single-tok nouns={len(held)}  seed={SEED}  N~{N}/tier")

    # default: content_addr_A (all tiers) + name_lookback (T1 ref). An optional argv
    # overrides the PRIMARY checkpoint (all tiers) — e.g. a conc_gate variant.
    primary = (sys.argv[1] if len(sys.argv) > 1 else "checkpoints/content_addr_A/best.pt")
    pname = primary.split("/")[-2] if "/" in primary else primary
    CKPTS = [(pname, primary), ("name_lookback", "checkpoints/name_lookback/best.pt")]

    # build the per-tier item lists ONCE (deterministic), shared across checkpoints
    tiers = [
        ("T1 surface",   gen_items(gen_tier1, held, N, SEED + 1)),
        ("T2 struct",    gen_items(gen_tier2, held, N, SEED + 2)),
        ("T3 pronoun",   gen_items(gen_tier3, held, N, SEED + 3)),
    ]
    t4_items, t4_samples = build_tier4(sp, held, allnouns, N, SEED + 4)
    tiers.append(("T4 natural", t4_items))

    # ── fairness: print 3 sample items per tier ──────────────────────────────────
    print("\n" + "=" * 78)
    print("SAMPLE ITEMS (3/tier) — verify the answer is NOT guessable from recency/"
          "frequency\n(cue names a NON-last subject; recency = the last-introduced "
          "object = a WRONG answer)")
    print("=" * 78)
    for name, items in tiers[:3]:
        print(f"\n[{name}]")
        for it in items[:3]:
            txt, c, r = sample_str(it)
            print(f"  {txt} ___")
            print(f"      answer={c!r}   recency-distractor={r!r}")
    print(f"\n[T4 natural]  (target = masked re-mention of a held-out noun)")
    for txt, c, r in t4_samples:
        show = txt if len(txt) <= 240 else "..." + txt[-237:]
        print(f"  {show} ___")
        print(f"      target={c!r}   recency-distractor={r!r}")

    # ── run each checkpoint; combined table ──────────────────────────────────────
    for ci, (cname, path) in enumerate(CKPTS):
        P.CKPT = path
        M.CKPT = path
        M.P.CKPT = path
        model, device, ema = P.load_model()
        meta = torch.load(path, map_location="cpu")
        use_ptr  = bool(meta.get("use_pointer", False))
        mem_copy = bool(meta.get("mem_copy", False))
        nlook    = bool(meta.get("name_lookback", False))
        assert not bool(meta.get("name_transport", False)), "name_transport must be OFF"
        assert not bool(meta.get("oracle_bind", False)), "oracle_bind must be OFF"

        # template baseline row = the STANDARD multi-eval, held-out mixed-k (reuse
        # probe_multi_eval.measure verbatim; documented reference ~74.2% 2AFC).
        train_objs = M.single_tok(sp, vocab["train"])
        base = M.measure(model, device, sp, held, 0, use_ptr, mem_copy,
                         oracle=False, name_transport=False, name_lookback=nlook)
        bc1, bc5, br1, bafc, bmc, bmr, blo, blr, bn = base

        # which tiers to run: all four for the PRIMARY checkpoint (index 0);
        # Tier 1 only for the name_lookback comparison row, per spec.
        run = tiers if ci == 0 else tiers[:1]

        print("\n" + "=" * 96)
        print(f"CHECKPOINT: {cname}  ({path})   ema_ce={ema:.4f}  "
              f"use_pointer={use_ptr} mem_copy={mem_copy} name_lookback={nlook}")
        print("=" * 96)
        print(f"{'tier':<16} {'cue/tgt-top1':>12} {'top5':>7} {'recency1':>9} "
              f"{'2AFC':>7} | {'readCue':>8} {'readRec':>8} | {'locObj':>7} {'locRec':>7} {'n':>5}")
        print("-" * 96)
        print(f"{'template base':<16} {pct(bc1):>12} {pct(bc5):>7} {pct(br1):>9} "
              f"{pct(bafc):>7} | {bmc:>8.3f} {bmr:>8.3f} | "
              f"{pct(blo):>7} {pct(blr):>7} {bn:>5}   (held-out mixed-k; ref 74.2% 2AFC)")
        print("-" * 96)
        for tname, items in run:
            r = measure_gen(model, device, sp, items, use_ptr, mem_copy, nlook)
            if r is None:
                print(f"{tname:<16} (no scorable items)")
                continue
            rc, rr = (f"{r['readCue']:>8.3f}" if r['readCue'] == r['readCue'] else "     nan"), \
                     (f"{r['readRec']:>8.3f}" if r['readRec'] == r['readRec'] else "     nan")
            print(f"{tname:<16} {pct(r['cue1']):>12} {pct(r['cue5']):>7} {pct(r['rec1']):>9} "
                  f"{pct(r['afc']):>7} | {rc} {rr} | "
                  f"{pct(r['locObj']):>7} {pct(r['locRec']):>7} {r['n']:>5}")
        print("-" * 96)
    print("\nLegend: cue/tgt-top5 = REAL selective binding (T4: target-top5) | recency1 = "
          "shortcut rate (want LOW)\n2AFC = P(answer>recency-distractor), chance 50% | "
          "readCue/readRec = retrieval read-mass at answer vs recency source\n"
          "locObj = lookback argmax hits clause subject | locRec = recall attn hits "
          "cue name (name-localization; NaN when no name-binding structure, e.g. T4)")


if __name__ == "__main__":
    main()
