"""
STATISTICAL binding-generation verdict — stable top-5 hit-rate over MANY
randomized prompts (the 4-prompt test is too noisy step-to-step).

Three batches, gold entity NEVER in the visible suffix (prefix ends at "the"):
  [A] in-template   : the trained RECALL templates, entities ball/hat/key/car +
                      the full training OBJECTS vocab.
  [B] unseen-type   : SAME templates but entity TYPES never bound in training
                      (lamp/spoon/rope/flag/sock/mug/leaf/coin) -> does the
                      retrieval head copy a token it never learned to decode?
  [C] novel-struct  : out-of-template sentence structures (relative clause,
                      apposition) over the training vocab -> structural general.

For each example we read the model's own next-token top-k and score whether the
gold entity is in top-1 / top-5, and its mean rank. n examples per batch.

Run: python3 test_binding_stats.py <ckpt> [n]
"""
import os, sys, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm
import probe_binding_linear as P

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/attn_super_full/best.pt"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 120
P.CKPT = CKPT

# training-vocab objects (single-token, from the synthetic OBJECTS set)
IN_OBJS    = ["ball", "hat", "key", "car", "drum", "boat", "duck", "book",
              "cup", "box", "bell", "bear", "cake"]
# entity TYPES never used in binding training (verified single ▁token)
UNSEEN_OBJS = ["lamp", "spoon", "rope", "flag", "sock", "mug", "leaf", "coin"]

FEMALE = P.FEMALE; MALE = P.MALE; ATTRS = P.ATTRS; VERBS = P.VERBS
INTRO  = P.INTRO; FILLER = P.FILLER; RECALL = P.RECALL_PREFIX
# out-of-template structures (entity introduced in a non-training frame)
NOVEL_INTRO = ["The {obj} that {name} found was {attr}.",
               "It was {name}'s {obj}, a {attr} one.",
               "{name} saw a {attr} {obj} on the way home.",
               "There was a {attr} {obj} next to {name}."]
NOVEL_RECALL = ["{Pron} picked up the", "Then {name} reached for the",
                "{name} looked at the", "Soon {Pron} grabbed the"]


def make(rng, objs, novel=False):
    if rng.random() < 0.5:
        name, pron, Pron = rng.choice(FEMALE), "she", "She"
    else:
        name, pron, Pron = rng.choice(MALE), "he", "He"
    obj = rng.choice(objs)
    f = dict(name=name, pron=pron, Pron=Pron, attr=rng.choice(ATTRS),
             obj=obj, verb=rng.choice(VERBS))
    intro = (rng.choice(NOVEL_INTRO) if novel else rng.choice(INTRO)).format(**f)
    parts = [intro]
    for _ in range(rng.randint(1, 3)):
        parts.append(rng.choice(FILLER).format(**f))
    parts.append((rng.choice(NOVEL_RECALL) if novel else rng.choice(RECALL)).format(**f))
    return " ".join(parts), obj


@torch.no_grad()
def batch(model, device, sp, rng, objs, n, novel=False):
    t1 = t5 = 0; ranks = []
    for _ in range(n):
        text, obj = make(rng, objs, novel)
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(inp, tau=0.1, hard=True)["logits"][0, -1].float()
        gold = sp.piece_to_id("▁" + obj)
        rank = (logits > logits[gold]).sum().item() + 1
        ranks.append(rank); t1 += rank == 1; t5 += rank <= 5
    return t1 / n, t5 / n, sum(ranks) / len(ranks), \
        sorted(ranks)[len(ranks) // 2]


def main():
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = P.load_model()
    rng = random.Random(0)
    print(f"[stats] ckpt={CKPT} ema_ce={ema} n={N}/batch\n")
    print(f"{'batch':<26} {'top1':>7} {'top5':>7} {'meanRank':>9} {'medRank':>8}")
    print("-" * 60)
    for tag, objs, nv in [("[A] in-template", IN_OBJS, False),
                          ("[C] novel-structure", IN_OBJS, True),
                          ("[B] unseen entity type", UNSEEN_OBJS, False)]:
        a1, a5, mr, md = batch(model, device, sp, rng, objs, N, nv)
        print(f"{tag:<26} {100*a1:>6.1f}% {100*a5:>6.1f}% {mr:>9.1f} {md:>8d}")
    print("-" * 60)
    print("top5 >> 0 => the bound entity is decoded into predictions (chance≈0.1%)")


if __name__ == "__main__":
    main()
