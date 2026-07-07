"""
FINAL LEDGER MEASUREMENT (Step 6) — do learned single-transfers COMPOSE to 2 hops?

EVAL-ONLY. No training, no new modules, no checkpoint modified. The write_policy /
write_policy_f2 checkpoints were trained on SINGLE ownership transfers only; here we
ask whether that learned machinery generalizes ZERO-SHOT to a 2-hop chain:

    "A had a ball. B had a car. C had a hat.
     B took the ball from A. C took the ball from B.  [D had a toy.]  Then C wanted the ___"

The traveling object X (ball) starts at A, moves A->B, then B->C. Recall cued by the
FINAL owner C should emit X. Faithful to the Step-6 generator: recipient-before-object
TRANSFER_FRAMES (so the backward name-lookback resolves the NEW owner at each write),
TRANSFER_INTRO intros, TRANSFER_RECALL open prompt ('the'), same FILLER/ATTRS/VERBS,
held-out (unseen) nouns. name_src=None asserted at inference; deterministic seed;
loader/flags from checkpoint meta.

Controls (all on the SAME chains, only the recall cue changes):
  final : cue C -> correct=X (transferred twice). AFC = P(X > C's own object).
  mid   : cue B (mid-chain, HAD X then passed it on) -> should NOT still retrieve X.
          keptAFC = P(B's own > X); staleTop1 = P(argmax==X)  (want LOW).
  origin: cue A (gave X away at hop 1) -> DOUBLE stale. staleTop1 = P(argmax==X).
Fairness: in HALF the items a decoy intro is added AFTER the 2nd transfer, so the
most-recent noun is the decoy, not X — X is then NOT recoverable by recency alone.
Single-hop reference (hops=1, cue recipient) is drawn from the SAME generator so the
2-hop final AFC has a within-run one-hop baseline to compose against.

readMass row (read_dist at recall) across every write on X plus owners' own objects:
  Xintro (A's origin write) / Xt1 (hop-1 write) / Xt2 (hop-2 write) / midOwn / finOwn.
Composition => final-owner read lands on Xt2 (X re-bound under C at the last hop).

The question: is 2-hop final-owner AFC near the 1-hop AFC (COMPOSES), near the product
of two hops (~p1^2, INDEPENDENT), or below it (staleness compounds SUPER-multiplicatively)?

Run: python3 probe_chain_eval.py [n]   (default n=120; ckpts fixed below)
"""
import os, sys, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P
import probe_multi_eval as PM
import train_attn_super as TA

N    = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SEED = 0
CKPTS = ["checkpoints/write_policy/best.pt",       # f=0.35 (strongest single-hop)
         "checkpoints/write_policy_f2/best.pt",     # f=0.2
         "checkpoints/content_addr_A/best.pt"]      # no-transfer baseline


def wid(sp, w):
    return sp.piece_to_id("▁" + w)


def occ(pcs, w):
    wl = w.lower()
    return [i for i, p in enumerate(pcs) if p.replace("▁", "").lower() == wl]


def build_chain(rng, objs, hops, decoy):
    """Build ONE chain story (prefix ending BEFORE the recall clause). Returns the
    prefix parts + metadata; the recall clause is appended per-cue by recall_text().
    owners[0]=origin A, owners[i] takes X from owners[i-1] at hop i, owners[hops]=final.
    Each non-origin owner gets an OWN object; A's own object IS the traveling X."""
    n_owners = hops + 1
    n_names  = n_owners + (1 if decoy else 0)
    n_obj    = 1 + hops + (1 if decoy else 0)              # X + one own per recipient + decoy
    if len(TA.FEMALE) + len(TA.MALE) < n_names or len(objs) < n_obj:
        return None
    names = rng.sample(TA.FEMALE + TA.MALE, n_names)
    owners = names[:n_owners]
    decoy_name = names[n_owners] if decoy else None
    osel = rng.sample(objs, n_obj)
    X = osel[0]
    own = {owners[i]: osel[i] for i in range(1, n_owners)}  # recipients' own objects
    decoy_obj = osel[1 + hops] if decoy else None

    parts = []
    parts.append(rng.choice(TA.TRANSFER_INTRO).format(
        name=owners[0], attr=rng.choice(TA.ATTRS), obj=X))     # A had a X
    for i in range(1, n_owners):                                # each recipient's own obj
        parts.append(rng.choice(TA.TRANSFER_INTRO).format(
            name=owners[i], attr=rng.choice(TA.ATTRS), obj=own[owners[i]]))
    if rng.random() < 0.3:
        pr = ("she", "She") if owners[0] in TA.FEMALE else ("he", "He")
        parts.append(rng.choice(TA.FILLER).format(pron=pr[0], Pron=pr[1]))
    for i in range(1, n_owners):                                # the transfer chain
        parts.append(rng.choice(TA.TRANSFER_FRAMES).format(
            recip=owners[i], giver=owners[i - 1], obj=X))
    if decoy:                                                    # fairness: break recency==X
        parts.append(rng.choice(TA.TRANSFER_INTRO).format(
            name=decoy_name, attr=rng.choice(TA.ATTRS), obj=decoy_obj))
    return dict(parts=parts, owners=owners, X=X, own=own, hops=hops,
                decoy=decoy, decoy_obj=decoy_obj, verb=rng.choice(TA.VERBS))


def recall_text(story, cue_name):
    return " ".join(story["parts"] +
                    ["Then {n} {v} the".format(n=cue_name, v=story["verb"])])


def resolve(story, sp, cue_name):
    """Full ids + all write positions for a given recall cue. None on bad tokenize
    (X must occur intro + hops times; positions must be found)."""
    text = recall_text(story, cue_name)
    pcs  = sp.encode(text, out_type=str)
    ids  = sp.EncodeAsIds(text)
    if len(ids) > TA.T_LEN:
        return None
    Xocc = occ(pcs, story["X"])
    if len(Xocc) != story["hops"] + 1:
        return None
    owners = story["owners"]
    own_pos = {}
    for nm, ob in story["own"].items():
        o = occ(pcs, ob)
        if not o:
            return None
        own_pos[nm] = o[0]
    return dict(ids=ids, pcs=pcs, X_intro=Xocc[0], X_t=Xocc,        # X_t[i]=hop-i write
                own_pos=own_pos, recall_pos=len(ids) - 1)


@torch.no_grad()
def emit_and_read(model, device, ids, use_ptr, mem_copy):
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    # name_src=None: the locator must find names unaided (asserted — nothing supplied).
    out = model(inp, tau=0.1, hard=True, use_pointer=use_ptr, subj_id=None,
                name_src=None)
    if use_ptr:
        logits = torch.log(out["pointer"]["P"][0, -1].float().clamp_min(1e-9))
    else:
        logits = out["logits"][0, -1].float()
    row = (out["sep_stats"]["read_dist"][0, -1].float()
           if mem_copy and out.get("sep_stats") is not None else None)
    return logits, row


def gen_items(sp, objs, hops, n, seed):
    """Deterministic list of resolvable chain stories (identical across checkpoints)."""
    rng = random.Random(seed)
    items = []
    guard = 0
    while len(items) < n and guard < n * 40:
        guard += 1
        decoy = (len(items) % 2 == 0)          # half decoy, half not
        st = build_chain(rng, objs, hops, decoy)
        if st is None:
            continue
        # require the final-owner resolve to tokenize cleanly (shared prefix)
        if resolve(st, sp, st["owners"][-1]) is None:
            continue
        items.append(st)
    return items


@torch.no_grad()
def eval_ckpt(ckpt, sp, objs, items1, items2, n):
    P.CKPT = ckpt
    model, device, ema = P.load_model()
    meta = torch.load(ckpt, map_location="cpu")
    use_ptr  = bool(meta.get("use_pointer", False))
    mem_copy = bool(meta.get("mem_copy", False))

    def score_final(items, hops):
        m = dict(n=0, top1=0, top5=0, afc=0, rec=0,
                 rXi=0.0, rXt1=0.0, rXt2=0.0, rMid=0.0, rFin=0.0)
        for st in items:
            fin = st["owners"][-1]
            r = resolve(st, sp, fin)
            if r is None:
                continue
            logits, row = emit_and_read(model, device, r["ids"], use_ptr, mem_copy)
            Xid  = wid(sp, st["X"])
            finown = st["own"][fin]
            oid  = wid(sp, finown)
            m["n"]   += 1
            m["top1"]+= logits.argmax().item() == Xid
            m["top5"]+= ((logits > logits[Xid]).sum().item() + 1) <= 5
            m["afc"] += int(logits[Xid] > logits[oid])
            if st["decoy"]:                                    # recency = decoy noun
                m["rec"] += logits.argmax().item() == wid(sp, st["decoy_obj"])
            if row is not None:
                m["rXi"]  += row[r["X_intro"]].item()
                m["rXt1"] += row[r["X_t"][1]].item()
                if hops >= 2:
                    m["rXt2"] += row[r["X_t"][2]].item()
                m["rFin"] += row[r["own_pos"][fin]].item()
                if hops >= 2:
                    mid = st["owners"][-2]
                    m["rMid"] += row[r["own_pos"][mid]].item()
        return m

    def score_stale(items, cue_idx):
        """cue owners[cue_idx]; report staleTop1=P(emit X) and (mid only) keptAFC."""
        m = dict(n=0, stale=0, keptAFC=0, keptTop5=0)
        for st in items:
            cue = st["owners"][cue_idx]
            r = resolve(st, sp, cue)
            if r is None:
                continue
            logits, _ = emit_and_read(model, device, r["ids"], use_ptr, mem_copy)
            Xid = wid(sp, st["X"])
            m["n"] += 1
            m["stale"] += logits.argmax().item() == Xid
            if cue in st["own"]:                               # mid owner has an own obj
                kid = wid(sp, st["own"][cue])
                m["keptAFC"]  += int(logits[kid] > logits[Xid])
                m["keptTop5"] += ((logits > logits[kid]).sum().item() + 1) <= 5
        return m

    one  = score_final(items1, 1)                              # single-hop reference
    two  = score_final(items2, 2)                              # 2-hop final owner
    mid  = score_stale(items2, cue_idx=1)                      # mid-chain owner (B)
    orig = score_stale(items2, cue_idx=0)                      # origin (A) double-stale
    return dict(ema=ema, one=one, two=two, mid=mid, orig=orig)


def pct(x, n): return 100.0 * x / n if n else float("nan")
def avg(x, n): return x / n if n else float("nan")


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    import json
    vocab = json.load(open("data/wide_vocab.json"))
    objs = PM.single_tok(sp, vocab["test"])                    # HELD-OUT nouns
    items1 = gen_items(sp, objs, 1, N, SEED)                   # single-hop
    items2 = gen_items(sp, objs, 2, N, SEED + 1)               # 2-hop chains

    print(f"\n[chain-eval] EVAL-ONLY 2-hop transfer composition. held-out nouns={len(objs)} "
          f"n(1hop)={len(items1)} n(2hop)={len(items2)}  seed={SEED}")
    print("[chain-eval] name_src=None at inference (locator unaided); deterministic; "
          "no checkpoint modified; zero-shot (trained on SINGLE transfers only).")
    print("[chain-eval] fairness: half the 2-hop items add a decoy intro AFTER hop-2 "
          "so the most-recent noun is NOT the traveling object.")

    # ── 3 samples ───────────────────────────────────────────────────────────────
    print("\n--- 3 sample 2-hop chains (recall cue = FINAL owner; correct = X) ---")
    for st in items2[:3]:
        fin = st["owners"][-1]
        print(f"  X={st['X']:<10} owners={'->'.join(st['owners'])}  decoy={st['decoy']}")
        print(f"    \"{recall_text(st, fin)} ___\"   (correct: {st['X']})")

    rows = [(c, eval_ckpt(c, sp, objs, items1, items2, N)) for c in CKPTS]

    # ── THE LEDGER TABLE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("2-HOP COMPOSITION LEDGER  (final-owner recall: does X survive two transfers?)")
    print("=" * 108)
    hdr = (f"{'checkpoint':<20} {'1hop AFC':>9} {'2hop AFC':>9} {'prod=1hop²':>11} "
           f"{'2hop t1/t5':>12} {'midStale':>9} {'midKeptAFC':>11} {'origStale':>10}")
    print(hdr); print("-" * 108)
    for c, r in rows:
        one, two, mid, orig = r["one"], r["two"], r["mid"], r["orig"]
        a1 = pct(one["afc"], one["n"]); a2 = pct(two["afc"], two["n"])
        prod = a1 * a1 / 100.0                                   # p1^2 (independence)
        name = c.split("/")[1]
        print(f"{name:<20} {a1:>8.1f}% {a2:>8.1f}% {prod:>10.1f}% "
              f"{pct(two['top1'],two['n']):>5.1f}/{pct(two['top5'],two['n']):<5.1f}% "
              f"{pct(mid['stale'],mid['n']):>8.1f}% {pct(mid['keptAFC'],mid['n']):>10.1f}% "
              f"{pct(orig['stale'],orig['n']):>9.1f}%")
    print("-" * 108)
    print("1hop AFC   = single-hop final-owner P(X > own)  (within-run one-hop baseline)")
    print("2hop AFC   = P(X > final-owner's own)   |   prod=1hop² = independence prediction")
    print("2hop t1/t5 = final-owner top1 / top5 emit of X (recall cued by C)")
    print("midStale   = P(mid owner B emits X; want LOW)  | midKeptAFC = P(B's own > X)")
    print("origStale  = P(origin A emits X; double-stale, want LOW)")
    print("COMPOSES if 2hop AFC ~ 1hop AFC ; INDEPENDENT if ~ prod ; SUPER-MULT if < prod")

    # ── readMass decomposition (final-owner recall) ──────────────────────────────
    print("\n" + "=" * 108)
    print("readMass @ FINAL-owner recall — where does the read land across X's writes?")
    print("=" * 108)
    print(f"{'checkpoint':<20} {'Xintro':>8} {'Xt1(hop1)':>10} {'Xt2(hop2)':>10} "
          f"{'midOwn':>8} {'finOwn':>8} | {'1hop Xt1':>9} {'1hop finOwn':>12}")
    print("-" * 108)
    for c, r in rows:
        two, one = r["two"], r["one"]
        n2, n1 = two["n"], one["n"]
        name = c.split("/")[1]
        print(f"{name:<20} {avg(two['rXi'],n2):>8.3f} {avg(two['rXt1'],n2):>10.3f} "
              f"{avg(two['rXt2'],n2):>10.3f} {avg(two['rMid'],n2):>8.3f} "
              f"{avg(two['rFin'],n2):>8.3f} | {avg(one['rXt1'],n1):>9.3f} "
              f"{avg(one['rFin'],n1):>12.3f}")
    print("-" * 108)
    print("COMPOSITION => mass on Xt2 (X re-bound under C at the LAST hop). Stale-at-source")
    print("=> mass stuck on Xintro/Xt1. Recency/own-shortcut => mass on finOwn.")
    print(f"\n[chain-eval] ema_ce: " +
          "  ".join(f"{c.split('/')[1]}={r['ema']:.3f}" for c, r in rows))


if __name__ == "__main__":
    main()
