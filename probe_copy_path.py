"""
Why does the copy fail? The localiser showed the entity is 100% present at its
OWN position but ~chance at the recall position -> RETRIEVAL failure. The copy
path is the gated lookback attention:  x_out_t = x_t + gate_t * o_proj(retrieved_t)
with retrieved_t = sum_{u<=t} attn[t,u] * v_u.

Three things must all be true for the entity to reach the recall token "the":
  (1) attend back   : attn[recall, source] is large (query finds the entity key)
  (2) value carries : o_proj has non-trivial norm (the copied value is added)
  (3) gate open     : gate_recall > 0

We measure all three at the RECALL position, averaged over the binding examples,
to localise the broken link.

Run:  python3 probe_copy_path.py [ckpt]
"""
import os, sys
os.environ["WANDB_MODE"] = "disabled"
import torch
import torch.nn.functional as F
import sentencepiece as spm

import probe_binding_linear as P
from probe_source_vs_recall import find_entity_pos

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/nomeanpool_run/best.pt"
P.CKPT = CKPT


@torch.no_grad()
def main():
    torch.manual_seed(P.SEED)
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = P.load_model()
    hl = model.hybrid_lookback
    d = hl.d
    exs = P.build_examples()

    cap = {}
    pre = hl.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0].detach()))

    mass_src, mass_self, mass_max, rank_src = [], [], [], []
    copy_rel, gate_rec = [], []
    n = 0
    for text, label in exs:
        pieces = sp.encode(text, out_type=str)
        pos = find_entity_pos(pieces, P.ENTITIES[label])
        if pos is None:
            continue
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        cap.clear()
        model(inp, tau=0.1, hard=True)
        x = cap["x"][0]                          # (T,d) input to lookback
        T = x.size(0); t = T - 1                 # recall position
        q = hl.q_proj(x[t:t+1])                  # (1,d)
        k = hl.k_proj(x)                         # (T,d)
        v = hl.v_proj(x)
        scores = (q @ k.t()).squeeze(0).float() / (d ** 0.5)   # (T,)
        scores[pos+1:] = float("-inf") if False else scores[pos+1:]  # keep causal below
        # causal: only u <= t are allowed (all of them here since t is last)
        attn = torch.softmax(scores, dim=-1)     # (T,)
        retrieved = (attn.unsqueeze(-1) * v).sum(0)            # (d,)
        gate = torch.sigmoid(hl.gate_mlp(x[t]).squeeze()).item()
        contrib = gate * hl.o_proj(retrieved)                  # what's added to x_t
        rel = (contrib.norm() / (x[t].norm() + 1e-6)).item()   # copied size vs residual

        mass_src.append(attn[pos].item())
        mass_self.append(attn[t].item())
        mass_max.append(attn.max().item())
        rank_src.append(int((attn > attn[pos]).sum().item()) + 1)   # 1 = top
        copy_rel.append(rel)
        gate_rec.append(gate)
        n += 1
    pre.remove()

    def mean(a): return sum(a) / len(a)
    print(f"\n[copy] ckpt={CKPT}  examples={n}  o_proj_norm={hl.o_proj.weight.norm():.3f}")
    print("-" * 64)
    print(f"  attn mass recall->SOURCE (entity pos) : {mean(mass_src):.4f}")
    print(f"  attn mass recall->SELF   (the)        : {mean(mass_self):.4f}")
    print(f"  attn mass recall->ARGMAX (peak)       : {mean(mass_max):.4f}")
    print(f"  avg rank of source among attended u   : {mean(rank_src):.1f}  "
          f"(1=source is the top-attended; seq len ~{n and '~'}{'?'})")
    print(f"  gate at recall (mean)                 : {mean(gate_rec):.3f}")
    print(f"  copied contribution / residual norm   : {mean(copy_rel):.4f}")
    print("-" * 64)
    ms, mx = mean(mass_src), mean(mass_max)
    if ms < 1.5 * (1.0 / 12):           # ~uniform-ish over a dozen tokens
        print("  => link (1) BROKEN: the recall query does NOT attend back to the")
        print("     entity source (mass ~ uniform). The QK match never learned to")
        print("     point 'the' at the bound entity. Copy can't start.")
    elif mean(copy_rel) < 0.05:
        print("  => link (2/3) WEAK: it attends to the source but the copied value")
        print("     is negligible vs the residual (o_proj/gate too small).")
    else:
        print("  => attends AND copies materially, yet entity not decodable at recall")
        print("     -> o_proj/v project the entity into a non-identity-preserving"
              " subspace.")


if __name__ == "__main__":
    main()
