"""
WHY do unseen entity types score 0%?  Disambiguate the two hypotheses:

  H1 RETRIEVAL didn't generalize: the supervised QK learned the SPECIFIC training
     objects' keys, so for a novel noun the recall query never matches its source
     -> attn[recall->source] is LOW for unseen types.

  H2 DECODE didn't generalize: the QK is token-agnostic (points at the recently
     introduced noun structurally), so it DOES copy the unseen entity to recall
     -> attn[recall->source] is HIGH, but the lm_head has no learned mapping from
        the copied vector to that token -> top5 still 0%.

We measure attn[recall->source] mass + source-is-argmax on the SAME prompt
distribution for trained vs unseen objects. (Entity is in the input here so the
source position is locatable; recall = last position = the "the".)

Run: python3 probe_unseen_why.py <ckpt>
"""
import os, sys, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P
import test_binding_stats as S

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/attn_super_full/best.pt"
P.CKPT = CKPT
N = 120


def src_pos(pieces, entity):
    e = entity.lower()
    for i, p in enumerate(pieces):
        if e in p.lower().replace("▁", ""):
            return i
    return None


@torch.no_grad()
def measure(model, device, sp, rng, objs):
    mass, argmax_hit, decode5, n = [], 0, 0, 0
    for _ in range(N):
        text, obj = S.make(rng, objs, novel=False)   # in-template
        pieces = sp.encode(text, out_type=str)
        s = src_pos(pieces, obj)
        if s is None:
            continue
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(inp, tau=0.1, hard=True)
        row = out["hybrid_stats"]["attn"][0, -1].float()   # recall(last) -> u
        mass.append(row[s].item())
        argmax_hit += (row.argmax().item() == s)
        # also the decode: is the entity in top5 of the next-token logits?
        logits = out["logits"][0, -1].float()
        gold = sp.piece_to_id("▁" + obj)
        decode5 += ((logits > logits[gold]).sum().item() + 1) <= 5
        n += 1
    m = torch.tensor(mass)
    return m.mean().item(), argmax_hit / n, decode5 / n, n


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = P.load_model()
    rng = random.Random(0)
    print(f"\n[why] ckpt={CKPT}  n={N}/group\n")
    print(f"{'group':<22} {'attn[rec->src]':>14} {'src=argmax':>11} {'decode top5':>12}")
    print("-" * 62)
    for tag, objs in [("trained objects", S.IN_OBJS),
                      ("UNSEEN types", S.UNSEEN_OBJS)]:
        am, ax, d5, n = measure(model, device, sp, rng, objs)
        print(f"{tag:<22} {am:>14.3f} {100*ax:>10.1f}% {100*d5:>11.1f}%")
    print("-" * 62)
    print("HIGH attn for UNSEEN + 0% decode => H2 (copy works, decode didn't"
          " generalize).\nLOW attn for UNSEEN => H1 (retrieval query memorized"
          " training tokens).")


if __name__ == "__main__":
    main()
