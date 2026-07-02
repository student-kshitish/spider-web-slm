"""
THE REAL VERDICT — does the model PREDICT the bound entity?

The recall linear probe showed the entity is ~90% linearly present at the head
input after attention-supervision training. That is NECESSARY but not SUFFICIENT:
the lm_head must actually DECODE it into the next-token distribution. Here we feed
each binding prompt (entity NOT in the visible suffix) and read the model's own
top-5 prediction for the next token.

The forward pass uses ONLY the trained weights — the attention-supervision aux is
a training-time gradient term, it never touches inference. So this is already the
"supervision OFF at inference" condition (Step 3a).

Sections:
  [A] in-template  : ball/hat/key/car, the 4 canonical prompts.
  [B] out-of-template generalization (Step 3b):
        - different sentence structure than the RECALL training templates
        - longer filler distance source->recall
        - entity TYPES never seen in binding training (lamp/spoon/rope/flag/sock
          are NOT in the synthetic OBJECTS list)

Run: python3 test_binding_generation.py <ckpt>
"""
import os, sys
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P     # reuse identical model load

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/attn_super_full/best.pt"
P.CKPT = CKPT

# (prompt, gold-entity, section-tag, note)
IN_TEMPLATE = [
    ("Lily had a red ball. She threw the",  "ball", "in-tmpl", ""),
    ("Sara had a blue hat. Later, Sara wore the", "hat", "in-tmpl", ""),
    ("Ben found a small key. Ben used the", "key", "in-tmpl", ""),
    ("Tom had a green car. Tom drove the", "car", "in-tmpl", ""),
]

# Step 3b — different structure / longer distance / unseen entity types.
GENERALIZE = [
    # different sentence structure (relative clause, "that"/"which", question)
    ("The ball that Lily found was red. She picked up the", "ball", "gen-struct", "rel-clause"),
    ("It was Tom's car, an old green one. Tom got into the", "car", "gen-struct", "apposition"),
    # longer source->recall distance (many fillers)
    ("Mia had a shiny bell. She went to the park. She played in the yard. "
     "She sang a happy song. It was a sunny day. Then Mia rang the", "bell", "gen-dist", "long"),
    ("Sam got a fluffy bear. He ran down the hill. He sat by the tree. "
     "Birds flew in the sky. He took a little nap. At last Sam hugged the", "bear", "gen-dist", "long"),
    # entity TYPES never in binding training (not in OBJECTS list)
    ("Lucy had a bright lamp. Lucy turned on the", "lamp", "gen-unseen", "unseen-type"),
    ("Max found a big spoon. Max held the", "spoon", "gen-unseen", "unseen-type"),
    ("Anna had a long rope. Anna pulled the", "rope", "gen-unseen", "unseen-type"),
    ("Leo got a red flag. Leo waved the", "flag", "gen-unseen", "unseen-type"),
]


@torch.no_grad()
def run_block(model, device, sp, items, title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    hits = 0
    for prompt, gold, tag, note in items:
        ids = sp.EncodeAsIds(prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(inp, tau=0.1, hard=True)
        logits = out["logits"][0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(5)
        gold_id = sp.piece_to_id("▁" + gold)
        rank = (probs > probs[gold_id]).sum().item() + 1
        in5 = gold_id in top_i.tolist()
        hits += in5
        toks = [sp.id_to_piece(int(i)).replace("▁", "") for i in top_i]
        top5 = "  ".join(f"{t}={p:.3f}" for t, p in zip(toks, top_p.tolist()))
        mark = "HIT " if in5 else "miss"
        extra = f" [{note}]" if note else ""
        print(f"\n[{mark}] gold={gold!r} (rank {rank}, p={probs[gold_id]:.3f}){extra}")
        print(f"       {prompt!r}")
        print(f"       top5: {top5}")
    print(f"\n  -> {title}: {hits}/{len(items)} gold-in-top5")
    return hits, len(items)


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = P.load_model()
    print(f"[gen] ckpt={CKPT} ema_ce={ema}")
    a_h, a_n = run_block(model, device, sp, IN_TEMPLATE,
                         "[A] IN-TEMPLATE — the real verdict (every prior arm: 0/4)")
    g_h, g_n = run_block(model, device, sp, GENERALIZE,
                         "[B] GENERALIZATION — out-of-template / long / unseen entity types")
    print(f"\n{'#'*78}")
    print(f"SUMMARY  in-template={a_h}/{a_n}   generalization={g_h}/{g_n}")
    print(f"{'#'*78}")


if __name__ == "__main__":
    main()
