"""
WHY does 'emit retrieved identity' not generalize?  Is the held-out identity
ABSENT at the recall site, or PRESENT-but-unreadable by the untied lm_head?

At the recall position we read the head-input vector h (final_norm output) and
rank the gold noun two ways, restricted to the 246-noun pool:
  EMIT  : via the trained lm_head        (logit = lm_head_row . h)   -- the actual output
  EMBSIM: via embedding similarity       (score = embed_row  . h)    -- a TIED head would use this

If gold ranks FAR better under EMBSIM than under EMIT for HELD-OUT nouns, the
copied identity IS present in embedding space; the untied output head is the
bottleneck, and weight-tying (or a copy/pointer readout) would let emit generalize.
If gold ranks badly under BOTH, the identity never reaches the head -> deeper.

Run: python3 probe_emit_why.py <ckpt> [n]
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


@torch.no_grad()
def measure(model, device, sp, objs, pool_ids, seed):
    rng = random.Random(seed)
    cap = {}
    h = model.final_norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o.detach()))
    pool = torch.tensor(pool_ids, device=device)
    W_head = model.lm_head.weight[pool].float()          # (P,d) untied output rows
    W_emb  = model.embed.weight[pool].float()            # (P,d) input embeddings
    emit_r, emb_r, emit5, emb5, n = [], [], 0, 0, 0
    pos = {tid: k for k, tid in enumerate(pool_ids)}
    for _ in range(N):
        text, obj = S.make(rng, objs, novel=False)
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        cap.clear(); model(inp, tau=0.1, hard=True)
        hv = cap["h"][0, -1].float()                      # head input at recall
        gi = pos[sp.piece_to_id("▁" + obj)]               # gold index in pool
        emit = W_head @ hv                                # (P,) actual emit logits
        emb  = W_emb  @ hv                                # (P,) tied-head hypothetical
        er = (emit > emit[gi]).sum().item() + 1
        br = (emb  > emb[gi]).sum().item() + 1
        emit_r.append(er); emb_r.append(br)
        emit5 += er <= 5;  emb5 += br <= 5;  n += 1
    h.remove()
    return (sum(emit_r) / n, emit5 / n, sum(emb_r) / n, emb5 / n)


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    vocab = json.load(open("data/wide_vocab.json"))
    model, device, ema = P.load_model()
    pool_ids = [sp.piece_to_id("▁" + w) for w in (vocab["train"] + vocab["test"])]
    print(f"\n[emit-why] ckpt={CKPT} n={N}/group  pool={len(pool_ids)} nouns\n")
    print(f"{'group':<16} {'EMIT rank':>10} {'EMIT top5':>10} | "
          f"{'EMBSIM rank':>12} {'EMBSIM top5':>12}")
    print("-" * 66)
    for tag, objs in [("TRAINED", vocab["train"]), ("HELD-OUT", vocab["test"])]:
        er, e5, br, b5 = measure(model, device, sp, objs, pool_ids, seed=0)
        print(f"{tag:<16} {er:>10.1f} {100*e5:>9.1f}% | {br:>12.1f} {100*b5:>11.1f}%")
    print("-" * 66)
    print("EMBSIM >> EMIT for HELD-OUT => identity is PRESENT in embedding space;")
    print("the untied lm_head is the bottleneck -> tying / copy-readout fixes emit.")


if __name__ == "__main__":
    main()
