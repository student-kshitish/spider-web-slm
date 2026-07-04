"""
T1 EMIT-COLLAPSE DECOMPOSITION — EVALUATION ONLY (no training, no new modules,
nothing modified). content_addr_A already HAS a copy readout (pointer/mem_copy).
On templates it emits held-out nouns well; on T1 (new verbs/frames) mixture top-5
falls 62%->35% while selection (2AFC ~69%, readCue>readRec) survives. Question:
WHICH stage of the copy pipeline breaks off-template?

The mem_copy emit pipeline at the recall position, per the forward (core/web.py):
  gate   lambda = sigmoid(copy_gate(x))      -> weights the GENERATIVE branch;
         (1-lambda) weights COPY.  lambda>0.5 = generative-dominant (copy suppressed).
  read   read_dist (sep_stats) selects source positions u; readCue = read_dist[cs],
         cs = the cued object's intro-token position (the token whose embedding is copied).
  decode retrieved = sum_u read_dist[u]*embed(id[u]); P_copy = softmax(scale * cos(retrieved, E)).
  mix    P = lambda*P_gen + (1-lambda)*P_copy.

We measure, per item at recall, on T1 vs template items (both HELD-OUT nouns, so the
only difference is surface/frame — noun novelty is held constant):
  1 GATE   : mean lambda[recall]; frac generative-dominant (lambda>0.5).
  2 READ   : readCue distribution; frac concentrated (mass>0.5 on source) vs diffuse.
  3 DECODE : among concentrated-read items, frac with correct noun in P_copy top-5
             (copy decode alone, before mixing) -> does decode work given a good read?
  4 ORACLE : force read one-hot onto the known source cs (eval-time recompute of
             P_copy from the model's OWN embedding table; lambda & P_gen untouched).
             Report P_copy-oracle top5, mixture-oracle top5, lambda. >=80% => emit
             machinery is fine and the T1 collapse is UPSTREAM (gate + read leak);
             <80% => identity is destroyed in the value/decode path off-template.
  5 CROSS-TAB: of the T1 mixture-top5 FAILURES, attribute each to the first broken
             stage (gate handed to gen / read diffuse / copy decode wrong / mix dilution).

Sanity: loader from meta, name_src=None asserted, mem_copy asserted, deterministic
seeds (identical T1 items to probe_offtemplate), no checkpoint modified.

Run:  python3 probe_emit_decompose.py
"""
import os, sys, json, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import torch.nn.functional as F
import sentencepiece as spm

import probe_binding_linear as P      # loader (reused)
import probe_multi_eval as M          # make_multi / tok_pos / single_tok (reused)
import probe_offtemplate as O         # gen_tier1 / gen_items (identical T1 items)
import train_attn_super as TA

SEED = 0
N    = 100
CKPT = "checkpoints/content_addr_A/best.pt"


def top5_hit(prob, idx):
    return ((prob > prob[idx]).sum().item() + 1) <= 5


def gen_template(rng, objs):
    """Standard multi-eval template item (name intros + filler + name-cued recall),
    HELD-OUT nouns — the in-distribution reference for the same held-out vocab."""
    k = rng.choice([2, 3])
    return M.make_multi(rng, objs, k)


@torch.no_grad()
def decompose(model, device, sp, items, scale, en):
    """Per-item recall-position stage decomposition. en = row-normalized embedding
    table (V,d), precomputed once. Returns list of per-item records (dicts)."""
    recs = []
    for first, cobj, robj, ent_pairs, cue_name in items:
        if isinstance(first, str):
            pieces = sp.encode(first, out_type=str)
            ids = sp.EncodeAsIds(first)
        else:
            ids = list(first); pieces = [sp.id_to_piece(i) for i in ids]
        cs = M.tok_pos(pieces, cobj)                     # source = cued object intro pos
        cid = sp.piece_to_id("▁" + cobj)
        if cs is None or cid == sp.unk_id():
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        name_src = None
        assert name_src is None, "name_src must be None at inference"
        out = model(inp, tau=0.1, hard=True, use_pointer=True,
                    subj_id=None, name_src=name_src)
        pt = out["pointer"]
        assert pt is not None and out.get("sep_stats") is not None, "need mem_copy pointer"
        r = len(ids) - 1                                 # recall position
        lam    = pt["lambda"][0, r].item()               # weight on GENERATIVE branch
        P_gen  = pt["P_gen"][0, r].float()
        P_copy = pt["P_copy"][0, r].float()
        P_mix  = pt["P"][0, r].float()
        read   = out["sep_stats"]["read_dist"][0, r].float()
        readCue = read[cs].item()

        # ── ORACLE-ADDRESS: one-hot read on cs -> retrieved = embed(source token);
        #    recompute P_copy from the model's OWN embedding table, remix with the
        #    model's OWN lambda & P_gen. No model state modified.
        retr = model.embed.weight[ids[cs]].float()       # embed(source token) == retrieved|onehot
        rn   = F.normalize(retr, dim=-1)
        P_copy_o = F.softmax(scale * (en @ rn), dim=-1)   # (V,)
        P_mix_o  = lam * P_gen + (1.0 - lam) * P_copy_o

        recs.append(dict(
            cid=cid, lam=lam, readCue=readCue,
            mix5=top5_hit(P_mix, cid),
            copy5=top5_hit(P_copy, cid),
            gen5=top5_hit(P_gen, cid),
            copy5_o=top5_hit(P_copy_o, cid),
            mix5_o=top5_hit(P_mix_o, cid),
        ))
    return recs


def agg(recs):
    n = len(recs)
    conc = [x for x in recs if x["readCue"] > 0.5]        # concentrated-read items
    return dict(
        n=n,
        mix5=sum(x["mix5"] for x in recs) / n,
        copy5=sum(x["copy5"] for x in recs) / n,
        gen5=sum(x["gen5"] for x in recs) / n,
        lam_mean=sum(x["lam"] for x in recs) / n,
        frac_gen=sum(x["lam"] > 0.5 for x in recs) / n,   # gate generative-dominant
        readCue_mean=sum(x["readCue"] for x in recs) / n,
        frac_conc=len(conc) / n,                          # read mass>0.5 on source
        copy5_given_conc=(sum(x["copy5"] for x in conc) / len(conc)) if conc else float("nan"),
        copy5_o=sum(x["copy5_o"] for x in recs) / n,
        mix5_o=sum(x["mix5_o"] for x in recs) / n,
    )


def crosstab(recs):
    """Of mixture-top5 FAILURES, attribute to the FIRST broken stage."""
    fails = [x for x in recs if not x["mix5"]]
    b = dict(gate=0, read=0, decode=0, mix=0)
    for x in fails:
        if x["lam"] > 0.5:            b["gate"] += 1       # handed to generative
        elif x["readCue"] <= 0.5:     b["read"] += 1       # read diffuse
        elif not x["copy5"]:          b["decode"] += 1     # copy decode wrong
        else:                         b["mix"] += 1        # all ok, mixture diluted it out
    return len(fails), b


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
    scale = float(getattr(model.cfg.model, "mem_copy_scale", 12.0))
    en = F.normalize(model.embed.weight.float(), dim=-1)  # (V,d), precomputed once
    print(f"[decompose] ckpt={CKPT} ema_ce={ema:.4f} mem_copy_scale={scale} "
          f"name_lookback={bool(meta.get('name_lookback'))} n={N}/group held-out nouns={len(held)}")

    # identical T1 items to probe_offtemplate (same generator + seed); template ref
    t1_items   = O.gen_items(O.gen_tier1, held, N, SEED + 1)
    tmpl_items = O.gen_items(gen_template, held, N, SEED + 9)

    tmpl = agg(decompose(model, device, sp, tmpl_items, scale, en))
    t1r  = decompose(model, device, sp, t1_items, scale, en)
    t1   = agg(t1r)

    def row(tag, a):
        return (f"{tag:<12} {100*a['mix5']:>7.1f}% {a['lam_mean']:>8.2f} "
                f"{100*a['frac_gen']:>8.1f}% {a['readCue_mean']:>8.2f} "
                f"{100*a['frac_conc']:>8.1f}% {100*a['copy5_given_conc']:>10.1f}% "
                f"{100*a['copy5']:>8.1f}% {100*a['gen5']:>7.1f}% {a['n']:>5}")

    print("\n" + "=" * 104)
    print("EMIT-STAGE DECOMPOSITION at recall  (content_addr_A; both groups HELD-OUT nouns)")
    print("=" * 104)
    print(f"{'group':<12} {'mixTop5':>8} {'lam[rec]':>8} {'%gen>.5':>9} "
          f"{'readCue':>8} {'%read>.5':>9} {'copy5|read>.5':>11} {'copy5':>8} {'gen5':>7} {'n':>5}")
    print("-" * 104)
    print(row("template", tmpl))
    print(row("T1 surface", t1))
    print("-" * 104)
    print("mixTop5 = final emit (the collapse) | lam[rec] = gate on GEN branch, %gen>.5 = copy suppressed")
    print("readCue = read mass on source, %read>.5 = concentrated | copy5|read>.5 = copy DECODE given a good read")
    print("copy5/gen5 = each branch ALONE in top5 (all items)")

    # ── ORACLE-ADDRESS ARM (T1): force read one-hot on the true source ──────────
    print("\n" + "=" * 104)
    print("ORACLE-ADDRESS ARM — force read one-hot on true source cs (eval-time recompute; model unmodified)")
    print("=" * 104)
    print(f"{'group':<12} {'lam[rec]':>8} {'Pcopy_oracle top5':>18} {'mixture_oracle top5':>20}")
    print("-" * 104)
    print(f"{'template':<12} {tmpl['lam_mean']:>8.2f} {100*tmpl['copy5_o']:>17.1f}% {100*tmpl['mix5_o']:>19.1f}%")
    print(f"{'T1 surface':<12} {t1['lam_mean']:>8.2f} {100*t1['copy5_o']:>17.1f}% {100*t1['mix5_o']:>19.1f}%")
    print("-" * 104)
    print(">=80% Pcopy_oracle => emit/decode machinery FINE with perfect addressing; "
          "T1 collapse is UPSTREAM (gate + read).")
    print("(mixture_oracle still bounded by lambda: even a perfect copy is down-weighted "
          "if the gate handed mass to gen.)")

    # ── CROSS-TAB: T1 mixture-top5 failures by first broken stage ───────────────
    nfail, b = crosstab(t1r)
    print("\n" + "=" * 104)
    print(f"T1 EMIT-FAILURE CROSS-TAB  (of {nfail} mixture-top5 failures / {t1['n']} items; "
          f"first broken stage)")
    print("=" * 104)
    tot = max(1, nfail)
    print(f"  GATE   (lambda>0.5, handed to generative)      : {b['gate']:>3}  ({100*b['gate']/tot:>5.1f}%)")
    print(f"  READ   (gate ok, read mass<=0.5, diffuse)      : {b['read']:>3}  ({100*b['read']/tot:>5.1f}%)")
    print(f"  DECODE (gate+read ok, copy top5 misses noun)   : {b['decode']:>3}  ({100*b['decode']/tot:>5.1f}%)")
    print(f"  MIX    (all stages ok, mixture diluted it out) : {b['mix']:>3}  ({100*b['mix']/tot:>5.1f}%)")
    print("-" * 104)
    print(f"  => T1 emit failures: {100*b['gate']/tot:.0f}% gate, {100*b['read']/tot:.0f}% read, "
          f"{100*b['decode']/tot:.0f}% decode, {100*b['mix']/tot:.0f}% mix-dilution")


if __name__ == "__main__":
    main()
