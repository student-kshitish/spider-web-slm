"""
Multi-entity SELECTIVE-binding eval — the decisive test that distinguishes REAL
cue-conditioned binding from a "copy the recently-introduced noun" shortcut.

Every single-entity test is ambiguous: with one entity, "bind the cued entity"
and "copy the only/most-recent noun" give the SAME answer, so 87% could be copy
without binding. Here each prompt has k=2..3 entities bound to distinct subjects,
and the recall cue NAMES a subject that is NOT the most-recent one:

    "Lily had a red ball. Tom had a blue car. Then Lily threw the ___"
     correct = ball (Lily's)      recency distractor = car (most recent)

We score, on the held-out (unseen) and trained vocab:
  cue-correct emit : model emits the CUE subject's object   -> REAL binding
  recency emit     : model emits the MOST-RECENT object     -> the shortcut
  2AFC cue>recency : logit(cue) > logit(recency)            -> selection (chance 50%)
And, for pointer/mem_copy models, the memory/attn read mass at the CUE source vs
the RECENCY source (does the retrieval itself select by cue?).

Verdict: cue-correct high AND >> recency -> selective binding. cue-correct ~ chance
and recency high -> copy works, binding does not.

Run: python3 probe_multi_eval.py <ckpt> [n]
"""
import os, sys, json, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P
import train_attn_super as TA

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mem_integrated/best.pt"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 120
P.CKPT = CKPT


def single_tok(sp, words):
    """keep only words that are exactly one '▁word' token (so piece_to_id works)."""
    out = []
    for w in words:
        pid = sp.piece_to_id("▁" + w)
        if pid != sp.unk_id() and sp.encode("a " + w, out_type=str)[-1] == "▁" + w:
            out.append(w)
    return out


def make_multi(rng, objs, k):
    """Returns (prefix_text ending at 'the', cue_obj, recency_obj). cue is chosen
    NON-last so the most-recent entity (recency) is always a wrong-answer
    distractor; a k=3 example also has a middle distractor."""
    names = rng.sample(TA.FEMALE + TA.MALE, k)
    chosen = rng.sample(objs, k)
    prons = [("she", "She") if nm in TA.FEMALE else ("he", "He") for nm in names]
    parts = []
    for i in range(k):
        parts.append("{name} had a {attr} {obj}.".format(
            name=names[i], attr=rng.choice(TA.ATTRS), obj=chosen[i]))
    for _ in range(rng.randint(1, 2)):
        parts.append(rng.choice(TA.FILLER).format(pron=prons[0][0], Pron=prons[0][1]))
    cue = rng.randrange(k - 1)              # 0..k-2  (NON-last)
    recency = k - 1                         # most-recent entity = distractor
    parts.append("Then {name} {verb} the".format(
        name=names[cue], verb=rng.choice(TA.VERBS)))
    # also return per-entity (object, subject-name) and the cued name so the
    # oracle can label subject token ids at the object intros + recall position
    return (" ".join(parts), chosen[cue], chosen[recency],
            list(zip(chosen, names)), names[cue])


def tok_pos(pieces, word):
    w = word.lower()
    for i, p in enumerate(pieces):
        if p.replace("▁", "").lower() == w:
            return i
    return None


def name_pos_before(pieces, sp, nm, before):
    """Position of NAME's first sub-piece (e.g. '▁Lily'/'▁Fin') occurring before
    `before` — matches the data generator's name-transport labeling. Used to gather
    the subject/cue-name embedding as the slot key at inference (name-transport)."""
    fp = sp.encode(nm, out_type=str)[0]
    cand = [i for i, p in enumerate(pieces) if p == fp and i < before]
    return cand[-1] if cand else None


@torch.no_grad()
def measure(model, device, sp, objs, seed, use_ptr, mem_copy, oracle=False,
            k_fixed=None, name_transport=False, name_lookback=False):
    """k_fixed=None -> mixed k in {2,3} (the headline). k_fixed=K -> every prompt
    has exactly K simultaneous entities, for the k-sweep that exposes the S=32
    modular-hash collision ceiling (distinct subjects approaching slot count).

    name_lookback: the Step-2 learned name locator. name_src is NOT built or passed
    (the whole point — located unaided at inference); instead we SCORE the lookback
    attention's own name-localization: does a[obj_pos] argmax onto that clause's
    subject name, and a[recall] onto the cue name? (labels used only for scoring)."""
    rng = random.Random(seed)
    cue1 = cue5 = rec1 = afc = 0
    m_cue = m_rec = 0.0
    loc_obj_hit = loc_obj_tot = 0        # name-loc: object-intro -> clause subject
    loc_rec_hit = 0                      # name-loc: recall -> cue name
    n = 0
    n_pool = len(TA.FEMALE) + len(TA.MALE)      # name pool (distinct subjects available)
    while n < N:
        k = k_fixed if k_fixed is not None else rng.choice([2, 3])
        if k > n_pool or k > len(objs):
            break                                # can't sample k distinct subjects/objects
        text, cobj, robj, ent_pairs, cue_name = make_multi(rng, objs, k)
        pieces = sp.encode(text, out_type=str)
        cs, rs = tok_pos(pieces, cobj), tok_pos(pieces, robj)
        if cs is None or rs is None:
            continue
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        subj_id = None
        if oracle:                              # subject token id at each obj intro + recall
            sid = torch.full((1, len(ids)), -1, dtype=torch.long, device=device)
            for ob, nm in ent_pairs:
                p = tok_pos(pieces, ob)
                if p is not None:
                    sid[0, p] = sp.EncodeAsIds(nm)[0]
            sid[0, len(ids) - 1] = sp.EncodeAsIds(cue_name)[0]   # recall position
            subj_id = sid
        name_src = None
        if name_transport:                      # gather NAME-token embedding as the slot key
            ns = torch.full((1, len(ids)), -1, dtype=torch.long, device=device)
            for ob, nm in ent_pairs:
                p = tok_pos(pieces, ob)                          # object-intro position
                if p is not None:
                    npos = name_pos_before(pieces, sp, nm, p)    # that clause's subject name
                    if npos is not None:
                        ns[0, p] = npos
            cpos = name_pos_before(pieces, sp, cue_name, len(ids) - 1)  # cue name at recall
            if cpos is not None:
                ns[0, len(ids) - 1] = cpos
            name_src = ns
        if name_lookback:
            # Step 2: the locator gets NO name positions — it must find them itself.
            assert name_src is None, "name_src must be disabled for name_lookback at inference"
        out = model(inp, tau=0.1, hard=True, use_pointer=use_ptr, subj_id=subj_id,
                    name_src=name_src)
        # NAME-LOCALIZATION (name_lookback only): score whether the lookback's own
        # attention points at the governing name. Labels are used ONLY here for
        # scoring — they never enter the forward above.
        if name_lookback and out.get("name_stats") is not None:
            a = out["name_stats"]["attn"][0].float()                 # (T,u)
            for ob, nm in ent_pairs:
                op = tok_pos(pieces, ob)                              # object-intro pos
                if op is None:
                    continue
                tp = name_pos_before(pieces, sp, nm, op)             # its clause subject name
                if tp is None:
                    continue
                loc_obj_tot += 1
                loc_obj_hit += int(a[op].argmax().item() == tp)
            rp = len(ids) - 1                                        # recall pos ("the")
            cp = name_pos_before(pieces, sp, cue_name, rp)           # cue name
            if cp is not None:
                loc_rec_hit += int(a[rp].argmax().item() == cp)
        if use_ptr:
            Pv = out["pointer"]["P"][0, -1].float()
            logits = torch.log(Pv.clamp_min(1e-9))
        else:
            logits = out["logits"][0, -1].float()
        cid = sp.piece_to_id("▁" + cobj)
        rid = sp.piece_to_id("▁" + robj)
        cue1 += logits.argmax().item() == cid
        cue5 += ((logits > logits[cid]).sum().item() + 1) <= 5
        rec1 += logits.argmax().item() == rid
        afc  += logits[cid] > logits[rid]
        # retrieval selectivity: read mass at cue source vs recency source
        row = (out["sep_stats"]["read_dist"][0, -1].float() if mem_copy
               else out["hybrid_stats"]["attn"][0, -1].float()
               if use_ptr and out.get("hybrid_stats") else None)
        if row is not None:
            m_cue += row[cs].item(); m_rec += row[rs].item()
        n += 1
    if n == 0:
        return (float("nan"),) * 8 + (0,)
    loc_obj = (loc_obj_hit / loc_obj_tot) if loc_obj_tot else float("nan")
    loc_rec = loc_rec_hit / n
    return (cue1 / n, cue5 / n, rec1 / n, afc / n, m_cue / n, m_rec / n,
            loc_obj, loc_rec, n)


@torch.no_grad()
def addr_diagnostics(model, device, sp, objs, seed, n_ex, name_lookback):
    """Step-3 self-organization diagnostics on the LEARNED slot routing
    (sep_stats write_w/read_w), with the hash target gone. Reports:
      - routeEntropy : mean H of write_w (entity intros) & read_w (recall), in
        nats. Lower = confident slot commitment (the thing --w_entropy sharpens);
        max = ln(S). Diffuse routing (names smeared over slots) shows high entropy.
      - slotAgree    : for the SAME name, does its write-side occurrence (cue
        subject's object-intro) route to the SAME argmax slot as its read-side
        occurrence (the recall)? This is what makes retrieval work WITHOUT a hash.
      - distinctSlots: #slots that ever win an argmax across labeled positions —
        detects collapse (few slots) vs healthy spread (up to min(#names, S)).
    Labels are used ONLY for scoring; name_src stays None in the forward."""
    rng = random.Random(seed)
    Hs, agree_hit, agree_tot, used = [], 0, 0, set()
    n = 0
    while n < n_ex:
        k = rng.choice([2, 3])
        text, cobj, robj, ent_pairs, cue_name = make_multi(rng, objs, k)
        pieces = sp.encode(text, out_type=str)
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        ns = None
        if name_lookback:
            assert ns is None, "name_src must be disabled at inference"
        out = model(inp, tau=0.1, hard=True, use_pointer=True, name_src=ns)
        ss = out.get("sep_stats")
        if ss is None or "write_w" not in ss:
            return None
        ww = ss["write_w"][0].float(); rw = ss["read_w"][0].float()   # (T,S)
        def H(p): return float(-(p.clamp_min(1e-9).log() * p).sum())
        rp = len(ids) - 1
        for ob, nm in ent_pairs:                       # entropy + slot usage at intros
            op = tok_pos(pieces, ob)
            if op is None:
                continue
            Hs.append(H(ww[op])); used.add(int(ww[op].argmax()))
        Hs.append(H(rw[rp])); used.add(int(rw[rp].argmax()))
        cw = tok_pos(pieces, cobj)                      # cue name: write vs read slot
        if cw is not None:
            agree_tot += 1
            agree_hit += int(ww[cw].argmax().item() == rw[rp].argmax().item())
        n += 1
    if n == 0 or not Hs:
        return None
    return (sum(Hs) / len(Hs),
            (agree_hit / agree_tot) if agree_tot else float("nan"),
            len(used), n)


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    vocab = json.load(open("data/wide_vocab.json"))
    train_objs = single_tok(sp, vocab["train"])
    test_objs  = single_tok(sp, vocab["test"])
    model, device, ema = P.load_model()
    meta = torch.load(CKPT, map_location="cpu")
    use_ptr  = bool(meta.get("use_pointer", False))
    mem_copy = bool(meta.get("mem_copy", False))
    oracle   = bool(meta.get("oracle_bind", False))
    ntrans   = bool(meta.get("name_transport", False))
    nlook    = bool(meta.get("name_lookback", False))
    caddr    = bool(meta.get("content_addr", False))
    print(f"\n[multi-eval] ckpt={CKPT} ema_ce={ema} n={N}/group "
          f"use_pointer={use_ptr} mem_copy={mem_copy} oracle_bind={oracle} "
          f"name_transport={ntrans} name_lookback={nlook} content_addr={caddr}")
    if nlook:
        print("[multi-eval] name_lookback: name_src DISABLED at inference "
              "(locator finds the name itself); reporting name-localization accuracy.")
    print(f"[multi-eval] single-tok nouns: train={len(train_objs)} "
          f"held-out={len(test_objs)} | cue=NON-recent, recency=distractor\n")
    loc_hdr = f" | {'locObj':>7} {'locRec':>7}" if nlook else ""
    print(f"{'group':<16} {'cue-top1':>9} {'cue-top5':>9} {'recency1':>9} "
          f"{'2AFC':>6} | {'readCue':>8} {'readRec':>8}{loc_hdr}")
    print("-" * (74 + (18 if nlook else 0)))
    for tag, objs in [("TRAINED", train_objs), ("HELD-OUT", test_objs)]:
        c1, c5, r1, afc, mc, mr, lo, lr, nn = measure(
            model, device, sp, objs, 0, use_ptr, mem_copy, oracle=oracle,
            name_transport=ntrans, name_lookback=nlook)
        loc_col = f" | {100*lo:>6.1f}% {100*lr:>6.1f}%" if nlook else ""
        print(f"{tag:<16} {100*c1:>8.1f}% {100*c5:>8.1f}% {100*r1:>8.1f}% "
              f"{100*afc:>5.1f}% | {mc:>8.3f} {mr:>8.3f}{loc_col}")
    print("-" * (74 + (18 if nlook else 0)))
    print("cue-top5 = REAL selective binding | recency1 = shortcut rate (want LOW)")
    print("2AFC = P(cue>recency), chance 50% | readCue/readRec = retrieval mass at "
          "cue vs recency source")
    if nlook:
        print("locObj = lookback a[obj] argmax hits clause subject | "
              "locRec = a[recall] argmax hits cue name (name-localization accuracy)")

    # ── k-SWEEP: cue accuracy vs. number of simultaneous entities ────────────────
    # The S=32 modular-hash caveat: subject->slot is subj_id % 32, so as the number
    # of DISTINCT simultaneous subjects grows toward the slot count, two subjects
    # collide onto one slot and binding degrades. This sweep is the key measurement
    # (not just the headline mixed-k cue-top5): watch cue-top5/2AFC fall as k rises.
    print(f"\n[k-sweep] cue accuracy vs #simultaneous entities (HELD-OUT; slots S=32)")
    loc_hdr = f" {'locObj':>7} {'locRec':>7}" if nlook else ""
    print(f"{'k':>3} {'cue-top1':>9} {'cue-top5':>9} {'recency1':>9} "
          f"{'2AFC':>6} | {'readCue':>8} {'readRec':>8}{loc_hdr} {'n':>5}")
    print("-" * (70 + (16 if nlook else 0)))
    for k in (2, 3, 4, 5, 6, 8, 10):
        res = measure(model, device, sp, test_objs, k, use_ptr, mem_copy,
                      oracle=oracle, k_fixed=k, name_transport=ntrans,
                      name_lookback=nlook)
        if res[-1] == 0:
            print(f"{k:>3}   (skipped: cannot sample {k} distinct subjects/objects)")
            continue
        c1, c5, r1, afc, mc, mr, lo, lr, nn = res
        loc_col = f" {100*lo:>6.1f}% {100*lr:>6.1f}%" if nlook else ""
        print(f"{k:>3} {100*c1:>8.1f}% {100*c5:>8.1f}% {100*r1:>8.1f}% "
              f"{100*afc:>5.1f}% | {mc:>8.3f} {mr:>8.3f}{loc_col} {nn:>5}")
    print("-" * 70)
    print("expect cue-top5 & 2AFC to fall as k -> S=32 (subject-slot collisions under subj%32)")

    # ── STEP-3 SELF-ORGANIZATION DIAGNOSTICS (content_addr) ──────────────────────
    # With the subj%S hash target dropped, is the routing still confident, does the
    # same name reach the same slot write-vs-read, and are slots spread (not collapsed)?
    if caddr:
        import math
        print(f"\n[addr-diag] content_addr self-organization (hash target REMOVED). "
              f"S={meta.get('write_mode','?')} slots; max entropy = ln(S).")
        print(f"{'group':<16} {'routeEntropy':>13} {'slotAgree':>10} {'distinctSlots':>14}")
        print("-" * 56)
        for tag, objs in [("TRAINED", train_objs), ("HELD-OUT", test_objs)]:
            d = addr_diagnostics(model, device, sp, objs, 0, N, nlook)
            if d is None:
                print(f"{tag:<16} (no sep_stats — not a separable-memory checkpoint)")
                continue
            ent, agree, ndist, nn = d
            print(f"{tag:<16} {ent:>13.3f} {100*agree:>9.1f}% {ndist:>14}")
        print("-" * 56)
        print("routeEntropy: lower = confident slot commitment (diffuse routing = high) | "
              "slotAgree: same name -> same slot write-vs-read (retrieval needs this) | "
              "distinctSlots: watch for collapse (few) vs healthy spread")


if __name__ == "__main__":
    main()
