"""
DECISIVE EVAL for Step 5 — binding UPDATE (ownership transfer / state tracking).
HELD-OUT nouns throughout. name_src=None at inference (asserted). No checkpoint
modified. Reuses the existing loader (probe_binding_linear.load_model), the
held-out vocab, the transfer story generator (train_attn_super.build_transfer_item,
for_training=False -> prompts end at 'the'), and probe_multi_eval for the plain
no-transfer baseline.

The question: when ownership transfers ("Lily had a ball. Tom had a car. Tom took
the ball from Lily."), does the memory RE-BIND the object to the NEW owner — and
does the OLD binding go STALE correctly?

FOUR columns (held-out):
  1 NEW-OWNER recall   : does Tom now retrieve the BALL (transferred), not his own
    car (the "no-update" shortcut)?  updateAFC = P(logit(ball) > logit(car)).
  2 STALENESS (old)    : does Lily still WRONGLY retrieve the ball?  staleTop1 =
    P(argmax == ball). Low = the old binding went stale (correct); high = stale
    binding persists. We also report what she retrieves.
  3 NON-INVOLVED       : a third subject, NOT in the transfer — did the re-bind
    corrupt its unrelated binding?  ownTop5 + AFC(own > transferred).
  4 NO-TRANSFER        : plain selective binding (no transfer clause). Does adding
    transfer data degrade plain binding from the content_addr_A ~74% baseline?

readMass decomposition at EACH recall (the key diagnostic): the read_dist mass the
recall places on the TRANSFER-write (object re-stored under the new owner) vs the
ORIGINAL intro-write (object first stored under the giver). WHICH write does the
read land on?

Run: python3 probe_transfer_eval.py <ckpt> [n]
"""
import os, sys, json, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P
import probe_multi_eval as PM
import train_attn_super as TA

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/transfer_A/best.pt"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 120
P.CKPT = CKPT


def new_ds(sp):
    ds = TA.StructuredMix.__new__(TA.StructuredMix)
    ds.sp = sp; ds.rng = random.Random(0); ds.n = 1000
    return ds


@torch.no_grad()
def emit_and_read(model, sp, device, ids, use_ptr, mem_copy):
    """Return (last-position logits over vocab, read_dist row at last pos or None)."""
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(inp, tau=0.1, hard=True, use_pointer=use_ptr, subj_id=None,
                name_src=None)
    if use_ptr:
        logits = torch.log(out["pointer"]["P"][0, -1].float().clamp_min(1e-9))
    else:
        logits = out["logits"][0, -1].float()
    row = (out["sep_stats"]["read_dist"][0, -1].float()
           if mem_copy and out.get("sep_stats") is not None else None)
    return logits, row


def wid(sp, w):
    return sp.piece_to_id("▁" + w)


@torch.no_grad()
def measure_transfer(model, device, sp, ds, objs, cue_mode, seed, use_ptr,
                     mem_copy):
    """Step-6 two-object-giver eval. cue_mode in {'new','old','noninvolved'};
    held-out prompts ending at 'the'. Emit metrics (meaning by mode) + readMass on
    the four writes: transfer-write, transferred obj's ORIGINAL giver-slot intro,
    the KEPT obj's giver-slot intro, and the recipient's OWN intro.

      new  : correct=objT(transferred); shortcut=recip-own(objC).  DECAY target.
             top1/5=objT, afc=P(objT>objC) 'updateAFC', shortcut=argmax==objC.
      old  : correct=objK(KEPT); stale=objT(transferred).          ERASE target.
             top1/5=objK 'kept', afc=P(objK>objT) 'keptAFC', shortcut=argmax==objT
             'staleTop1' (want LOW).
      nonv : correct=objD(own); corruptor=objT. afc=P(own>objT), shortcut=corrupt.
    """
    rng = random.Random(seed)
    n = 0
    top1 = top5 = afc = shortcut = 0
    m_tr = m_origT = m_kept = m_own = 0.0
    def occ(pcs, w):
        wl = w.lower()
        return [i for i, p in enumerate(pcs) if p.replace("▁", "").lower() == wl]
    while n < N:
        k = 3 if cue_mode == "noninvolved" else rng.choice([2, 3])
        if k + 1 > len(objs):
            break
        it = ds.build_transfer_item(rng, objs, cue_mode, True, k,
                                    for_training=False)
        if it is None:
            continue
        ids = it["ids"]
        if len(ids) > TA.T_LEN:
            continue
        tp = it["transfer_pos"]; iT = it["intro_T_pos"]; iK = it["intro_K_pos"]
        ro = it["recip_own_pos"]
        if tp is None:
            continue
        logits, row = emit_and_read(model, sp, device, ids, use_ptr, mem_copy)
        objT, objK, cue_obj = it["objT"], it["objK"], it["cue_obj"]
        recip_own = it["recip_obj"]
        tTid, kKid, oid, cid = (wid(sp, objT), wid(sp, objK),
                                wid(sp, recip_own), wid(sp, cue_obj))
        if cue_mode == "new":
            correct, other = tTid, oid
        elif cue_mode == "old":
            correct, other = kKid, tTid              # kept vs transferred
        else:
            correct, other = cid, tTid               # own vs transferred (corruptor)
        top1 += logits.argmax().item() == correct
        top5 += ((logits > logits[correct]).sum().item() + 1) <= 5
        afc  += logits[correct] > logits[other]
        shortcut += logits.argmax().item() == other
        if row is not None:
            m_tr   += row[tp].item()
            m_origT += row[iT].item()
            m_kept += row[iK].item()
            m_own  += row[ro].item() if ro is not None else 0.0
        n += 1
    if n == 0:
        return None
    return dict(n=n, top1=top1 / n, top5=top5 / n, afc=afc / n,
                shortcut=shortcut / n, readTransfer=m_tr / n,
                readOrigT=m_origT / n, readKept=m_kept / n, readRecipOwn=m_own / n)


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    vocab = json.load(open("data/wide_vocab.json"))
    test_objs = PM.single_tok(sp, vocab["test"])          # HELD-OUT nouns
    model, device, ema = P.load_model()
    meta = torch.load(CKPT, map_location="cpu")
    use_ptr  = bool(meta.get("use_pointer", False))
    mem_copy = bool(meta.get("mem_copy", False))
    nlook    = bool(meta.get("name_lookback", False))
    transfer = bool(meta.get("transfer", False))
    wdecay   = float(meta.get("write_decay", 1.0))
    erase    = bool(meta.get("erase", False))
    tfrac    = meta.get("transfer_frac", None)
    ds = new_ds(sp)
    print(f"\n[transfer-eval] ckpt={CKPT} ema_ce={ema} n={N}/col  use_pointer={use_ptr} "
          f"mem_copy={mem_copy} name_lookback={nlook}")
    print(f"[transfer-eval] STEP-6 policies: write_decay(γ)={wdecay} erase={erase} "
          f"transfer={transfer} transfer_frac={tfrac}")
    print(f"[transfer-eval] HELD-OUT nouns={len(test_objs)}  name_src=None at inference "
          f"(both lookbacks locate names unaided)")
    if not transfer:
        print("[transfer-eval] WARNING: checkpoint meta transfer=False — running as a baseline.")

    # ── columns 1-3: transfer prompts ──────────────────────────────────────────
    new = measure_transfer(model, device, sp, ds, test_objs, "new", 0, use_ptr, mem_copy)
    old = measure_transfer(model, device, sp, ds, test_objs, "old", 1, use_ptr, mem_copy)
    non = measure_transfer(model, device, sp, ds, test_objs, "noninvolved", 2, use_ptr, mem_copy)

    print("\n" + "=" * 90)
    print("EMIT (held-out).  Step-5 refs: new updateAFC 55%, staleTop1 55%")
    print("=" * 90)
    print(f"{'column':<22} {'metric':<38} {'value':>9}  {'n':>4}")
    print("-" * 90)
    print(f"{'1 NEW-OWNER':<22} {'top1 transferred (re-bind)':<38} "
          f"{100*new['top1']:>8.1f}%  {new['n']:>4}")
    print(f"{'(DECAY target)':<22} {'top5 transferred':<38} {100*new['top5']:>8.1f}%")
    print(f"{'':<22} {'updateAFC P(transferred > recip-own)':<38} {100*new['afc']:>8.1f}%")
    print(f"{'':<22} {'no-update shortcut (emits recip-own)':<38} {100*new['shortcut']:>8.1f}%")
    print("-" * 90)
    print(f"{'2 STALENESS (old)':<22} {'keptTop5 (retrieves KEPT obj)':<38} "
          f"{100*old['top5']:>8.1f}%  {old['n']:>4}")
    print(f"{'(ERASE target)':<22} {'keptAFC P(kept > transferred)':<38} {100*old['afc']:>8.1f}%")
    print(f"{'':<22} {'staleTop1 (emits TRANSFERRED; want LOW)':<38} {100*old['shortcut']:>8.1f}%")
    print("-" * 90)
    print(f"{'3 NON-INVOLVED':<22} {'own-obj top5 (survived)':<38} "
          f"{100*non['top5']:>8.1f}%  {non['n']:>4}")
    print(f"{'':<22} {'AFC P(own > transferred)':<38} {100*non['afc']:>8.1f}%")
    print(f"{'':<22} {'corruption (emits transferred)':<38} {100*non['shortcut']:>8.1f}%")
    print("-" * 90)

    # ── readMass decomposition ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("readMass DECOMPOSITION  — WHERE the recall's read_dist lands (4 writes)")
    print("=" * 90)
    print(f"{'column':<22} {'readTransfer':>12} {'readOrigT':>10} {'readKept':>9} "
          f"{'readRecipOwn':>13}")
    print("-" * 90)
    for tag, r in [("1 NEW-OWNER", new), ("2 STALENESS (old)", old),
                   ("3 NON-INVOLVED", non)]:
        print(f"{tag:<22} {r['readTransfer']:>12.3f} {r['readOrigT']:>10.3f} "
              f"{r['readKept']:>9.3f} {r['readRecipOwn']:>13.3f}")
    print("-" * 90)
    print("readTransfer  = transfer-write (obj re-stored under NEW owner)")
    print("readOrigT     = TRANSFERRED obj's ORIGINAL intro-write (in the giver slot; stale src)")
    print("readKept      = KEPT obj's intro-write (in the giver slot)")
    print("readRecipOwn  = recipient's OWN pre-transfer intro")
    print("  NEW-OWNER correct  : read lands on TRANSFER-write (not readRecipOwn) — DECAY.")
    print("  STALENESS correct  : read lands on readKept, NOT readOrigT — ERASE suppresses objT.")

    # ── column 4: no-transfer baseline (plain selective binding) ────────────────
    print("\n" + "=" * 90)
    print("4 NO-TRANSFER BASELINE — plain selective binding (does transfer data degrade it?)")
    print("=" * 90)
    c1, c5, r1, afc, mc, mr, lo, lr, nn = PM.measure(
        model, device, sp, test_objs, 0, use_ptr, mem_copy,
        name_lookback=nlook)
    print(f"held-out  cue-top5={100*c5:.1f}%  2AFC(cue>recency)={100*afc:.1f}%  "
          f"recency1={100*r1:.1f}%  readCue={mc:.3f} readRec={mr:.3f}  n={nn}")
    print(f"  reference: content_addr_A held-out 2AFC = 74.2% (Step-3 baseline). "
          f"Compare: {100*afc:.1f}%  -> "
          f"{'NO regression' if afc >= 0.70 else 'DEGRADED'}")


if __name__ == "__main__":
    main()
