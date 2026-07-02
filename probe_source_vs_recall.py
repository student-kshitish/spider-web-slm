"""
ROOT-CAUSE LOCALISER — is the entity destroyed at its OWN position, or merely
never COPIED to the recall position?

The 4-way linear probe (probe_binding_linear.py) reads only the RECALL position
("...the") and finds ~30% (chance) in every config. That tells us the entity is
absent THERE but not WHY. Two very different causes:

  (b) RETRIEVAL failure : the entity IS linearly present at its OWN position
      (e.g. the "ball" token) all the way to the head, but the routing/lookback
      never copies it forward to the recall site. Fix = the copy/attention path.

  (a) REPRESENTATION failure : the entity is destroyed even at its own position
      as it passes through the rings (chaos routing / ring-mean / per-hop RMSNorm
      scrub the token-specific direction). Nothing downstream could ever copy
      what no longer exists. Fix = the substrate itself.

Same examples, same 4 stages (embed+RoPE / post-rings / post-attention /
head-input), same held-out linear probe as probe_binding_linear.py. The only
change: we ALSO read the hidden state at the entity's OWN token position (the
word IS in the input here), and compare the two trajectories.

  SOURCE high at A but DECAYS by D  -> (a) substrate destroys it locally.
  SOURCE high through D, RECALL low  -> (b) it survives locally; copy fails.

Run:  python3 probe_source_vs_recall.py [ckpt]
"""
import os, sys
os.environ["WANDB_MODE"] = "disabled"

import torch
import sentencepiece as spm

import probe_binding_linear as P   # reuse IDENTICAL model load / examples / probe

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/nomeanpool_run/best.pt"
P.CKPT = CKPT          # load_model reads this module global


def find_entity_pos(pieces, entity):
    """First token index whose subword piece contains the entity word."""
    e = entity.lower()
    for i, p in enumerate(pieces):
        if e in p.lower().replace("▁", ""):
            return i
    return None


@torch.no_grad()
def extract_both(model, device, sp, examples):
    cap = {}
    h = [
        model.rope.register_forward_hook(
            lambda m, i, o: cap.__setitem__("A", o.detach())),
        model.hybrid_lookback.register_forward_pre_hook(
            lambda m, i: cap.__setitem__("B", i[0].detach())),
        model.hybrid_lookback.register_forward_hook(
            lambda m, i, o: cap.__setitem__("C", o[0].detach())),
        model.final_norm.register_forward_hook(
            lambda m, i, o: cap.__setitem__("D", o.detach())),
    ]
    src = {s: [] for s in "ABCD"}
    rec = {s: [] for s in "ABCD"}
    labels = []
    skipped = 0
    for text, label in examples:
        pieces = sp.encode(text, out_type=str)
        pos = find_entity_pos(pieces, P.ENTITIES[label])
        if pos is None:
            skipped += 1
            continue
        ids = sp.EncodeAsIds(text)
        assert len(ids) == len(pieces)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        cap.clear()
        model(inp, tau=0.1, hard=True)
        for s in "ABCD":
            src[s].append(cap[s][0, pos].float().cpu())
            rec[s].append(cap[s][0, -1].float().cpu())
        labels.append(label)
    for hk in h:
        hk.remove()
    src = {s: torch.stack(v) for s, v in src.items()}
    rec = {s: torch.stack(v) for s, v in rec.items()}
    return src, rec, torch.tensor(labels), skipped


def eval_stage(feats, labels):
    N = len(labels); n_te = int(N * P.TEST_FRAC)
    out = {}
    for s in "ABCD":
        tr_a, te_a = [], []
        for k in range(P.N_SPLITS):
            g = torch.Generator().manual_seed(100 + k)
            perm = torch.randperm(N, generator=g)
            te, tr = perm[:n_te], perm[n_te:]
            a, b = P.train_probe(feats[s][tr], labels[tr], feats[s][te], labels[te])
            tr_a.append(a); te_a.append(b)
        out[s] = (sum(tr_a) / P.N_SPLITS, sum(te_a) / P.N_SPLITS)
    return out


def main():
    torch.manual_seed(P.SEED)
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = P.load_model()
    exs = P.build_examples()
    print(f"[loc] ckpt={CKPT} ema_ce={ema} examples={len(exs)}", flush=True)

    src, rec, labels, skipped = extract_both(model, device, sp, exs)
    print(f"[loc] usable={len(labels)} skipped(entity-not-found)={skipped}", flush=True)

    s_res = eval_stage(src, labels)
    r_res = eval_stage(rec, labels)

    name = {"A": "embed+RoPE", "B": "post-rings", "C": "post-attention",
            "D": "head-input"}
    print(f"\n{'stage':<16} {'SOURCE pos (entity token)':>26}   "
          f"{'RECALL pos (the)':>22}   chance=25%")
    print(f"{'':<16} {'train':>12} {'test':>12}   {'train':>10} {'test':>10}")
    print("-" * 72)
    for s in "ABCD":
        st, se = s_res[s]; rt, re = r_res[s]
        print(f"{name[s]:<16} {100*st:>11.1f}% {100*se:>11.1f}%   "
              f"{100*rt:>9.1f}% {100*re:>9.1f}%", flush=True)
    print("-" * 72)

    sA, sD = s_res["A"][1], s_res["D"][1]
    rD = r_res["D"][1]
    print(f"\nSOURCE A->D: " + " ".join(f"{s}={100*s_res[s][1]:.0f}%" for s in "ABCD"))
    print(f"RECALL A->D: " + " ".join(f"{s}={100*r_res[s][1]:.0f}%" for s in "ABCD"))
    print()
    if sD > 0.60 and rD < 0.40:
        print("VERDICT => (b) RETRIEVAL failure: entity survives at its OWN position")
        print("           through the head, but is never copied to the recall site.")
        print("           Fix the copy/lookback path, NOT the substrate.")
    elif sA > 0.60 and sD < 0.45:
        print("VERDICT => (a) REPRESENTATION failure: entity is present at embed but")
        print(f"           DECAYS at its own position by the head ({100*sA:.0f}%->{100*sD:.0f}%).")
        print("           The substrate (chaos routing / ring-mean / RMSNorm) scrubs")
        print("           the token-specific direction. Fix the substrate.")
    elif sA < 0.60:
        print("VERDICT => entity not even recoverable at its OWN position at embed")
        print("           (unexpected — check tokenisation / probe).")
    else:
        print("VERDICT => mixed; read the trajectories above.")


if __name__ == "__main__":
    main()
