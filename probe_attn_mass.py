"""
Clean attn[recall->source] mass on HELD-OUT probe examples (entity NOT in input;
prefix ends at "the"). Direct test of whether the retrieval query formed.

For each example we locate the source position (where the entity was introduced
in the intro clause) and read the lookback attention from the LAST position (the
recall "the") to that source. Reports mean mass and mean rank-of-source.

Run: python3 probe_attn_mass.py <ckpt>
"""
import os, sys
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/attn_super/best.pt"
P.CKPT = CKPT


def src_pos(pieces, entity):
    e = entity.lower()
    for i, p in enumerate(pieces):
        if e in p.lower().replace("▁", ""):
            return i
    return None


@torch.no_grad()
def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = P.load_model()
    exs = P.build_examples()
    masses, ranks, n = [], [], 0
    for text, label in exs:
        pieces = sp.encode(text, out_type=str)
        s = src_pos(pieces, P.ENTITIES[label])
        if s is None:
            continue
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(inp, tau=0.1, hard=True)
        attn = out["hybrid_stats"]["attn"].float()      # (1,T,u)
        row = attn[0, -1]                               # recall(last) -> u
        masses.append(row[s].item())
        ranks.append((row > row[s]).sum().item() + 1)   # 1 = top
        n += 1
    masses = torch.tensor(masses); ranks = torch.tensor(ranks, dtype=torch.float)
    print(f"\n[attn-mass] ckpt={CKPT}  n={n}")
    print(f"[attn-mass] attn[recall->source] mass: mean={masses.mean():.3f} "
          f"median={masses.median():.3f}")
    print(f"[attn-mass] rank-of-source         : mean={ranks.mean():.1f} "
          f"median={int(ranks.median())}  (1=argmax)")
    print(f"[attn-mass] frac source==argmax    : {(ranks==1).float().mean():.3f}")


if __name__ == "__main__":
    main()
