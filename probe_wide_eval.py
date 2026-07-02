"""
Step 3 — decisive generalization test. Measure, on the SAME prompt distribution,
the retrieval+decode metrics for TRAINED nouns vs HELD-OUT nouns the model never
saw during training. n=120/group.

  attn[recall->source] mass   (narrow held-out baseline: 0.23; trained: 0.99)
  source = argmax %           (narrow held-out: 24%; trained: 100%)
  decode top-5 %              (narrow held-out: 0%;  trained: ~56%)

Run: python3 probe_wide_eval.py <ckpt> [n]
"""
import os, sys, json, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P
import test_binding_stats as S

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/wide_vocab/best.pt"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 120
P.CKPT = CKPT


def src_pos(pieces, entity):
    e = entity.lower()
    for i, p in enumerate(pieces):
        if e in p.lower().replace("▁", ""):
            return i
    return None


@torch.no_grad()
def measure(model, device, sp, objs, pool_ids, seed, use_ptr=False, mem_copy=False):
    """pool_ids: token ids of the FULL noun pool (train+test). We also score the
    gold's rank restricted to that pool (does the model pick the bound noun among
    plausible objects?) and a 2AFC vs a random other pool noun — both isolate
    'decode present but diffuse over 200 objects' from 'decode absent'."""
    rng = random.Random(seed)
    pool = torch.tensor(pool_ids, device=device)
    mass, ax, d5, pool5, afc, ranks, n = [], 0, 0, 0, 0, [], 0
    while n < N:
        text, obj = S.make(rng, objs, novel=False)
        pieces = sp.encode(text, out_type=str)
        s = src_pos(pieces, obj)
        if s is None:
            continue
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(inp, tau=0.1, hard=True, use_pointer=use_ptr)
        # retrieval distribution feeding the copy: the orbital MEMORY read for
        # mem_copy, else the hybrid lookback attn. (mass/argmax @ source position.)
        row = (out["sep_stats"]["read_dist"][0, -1].float() if mem_copy
               else out["hybrid_stats"]["attn"][0, -1].float())
        mass.append(row[s].item())
        ax += (row.argmax().item() == s)
        # pointer arm: read the prediction from the copy/gen mixture P (as
        # log-probs so rank/top-5/2AFC comparisons are identical to the logit
        # path). OFF arm: the usual generative logits.
        if use_ptr:
            P = out["pointer"]["P"][0, -1].float()
            logits = torch.log(P.clamp_min(1e-9))
        else:
            logits = out["logits"][0, -1].float()
        gold = sp.piece_to_id("▁" + obj)
        d5 += ((logits > logits[gold]).sum().item() + 1) <= 5
        # rank among the noun pool only
        pl = logits[pool]
        pool_rank = (pl > logits[gold]).sum().item() + 1
        pool5 += pool_rank <= 5
        ranks.append(pool_rank)
        # 2AFC: gold vs a random DIFFERENT pool noun
        other = gold
        while other == gold:
            other = pool_ids[rng.randrange(len(pool_ids))]
        afc += logits[gold] > logits[other]
        n += 1
    return (sum(mass) / n, ax / n, d5 / n, pool5 / n,
            sum(ranks) / n, afc / n)


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    vocab = json.load(open("data/wide_vocab.json"))
    assert not (set(vocab["train"]) & set(vocab["test"])), "vocab overlap!"
    model, device, ema = P.load_model()
    meta = torch.load(CKPT, map_location="cpu")
    use_ptr  = bool(meta.get("use_pointer", False))
    mem_copy = bool(meta.get("mem_copy", False))
    csrc = ("orbital memory read_dist + embed-similarity" if mem_copy
            else "hybrid attn scatter" if use_ptr else "logits")
    print(f"\n[wide-eval] ckpt={CKPT} ema_ce={ema} n={N}/group use_pointer={use_ptr}"
          f" mem_copy={mem_copy}  (copy source = {csrc})")
    print(f"[wide-eval] train nouns={len(vocab['train'])} held-out={len(vocab['test'])}"
          f" overlap=0\n")
    pool_ids = [sp.piece_to_id("▁" + w) for w in (vocab["train"] + vocab["test"])]
    print(f"{'group':<16} {'attn':>6} {'argmax':>7} {'dec5(full)':>11} "
          f"{'dec5(pool)':>11} {'poolRank':>9} {'2AFC':>6}")
    print("-" * 74)
    for tag, objs in [("TRAINED nouns", vocab["train"]),
                      ("HELD-OUT nouns", vocab["test"])]:
        am, ax, d5, p5, mr, afc = measure(model, device, sp, objs, pool_ids,
                                          seed=0, use_ptr=use_ptr, mem_copy=mem_copy)
        print(f"{tag:<16} {am:>6.3f} {100*ax:>6.1f}% {100*d5:>10.1f}% "
              f"{100*p5:>10.1f}% {mr:>9.1f} {100*afc:>5.1f}%")
    print("-" * 74)
    print("narrow held-out baseline: attn 0.23 | argmax 24% | dec5(full) 0%")
    print("dec5(pool)=rank<=5 among the 246 nouns | poolRank=mean rank among nouns"
          " | 2AFC=gold>random other noun (chance 50%)")


if __name__ == "__main__":
    main()
