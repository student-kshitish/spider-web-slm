"""
Step 1 — build a WIDE, DISJOINT entity vocabulary for the generalization test.

Assemble a large pool of concrete nouns, keep only those that are a SINGLE
SentencePiece token ('▁noun' -> 1 piece) so the recall target is one token, then
split ~80/20 into train / held-out with ZERO overlap. Saved to data/wide_vocab.json.
"""
import json, random
import sentencepiece as spm

sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")

# ~320 concrete-noun candidates (objects / animals / food / household / nature)
CANDIDATES = """
ball hat cat dog book cup kite doll box drum frog fish cake key shoe car bell duck
bear boat lamp spoon rope flag sock mug leaf coin ring sword crown wand broom
chair table bed door window roof wall floor clock plate bowl fork knife dish pot pan
brush comb towel soap toy block train plane truck bike scooter wagon sled cart
apple banana orange grape lemon peach pear plum cherry berry melon corn bean pea
carrot potato tomato onion bread butter cheese egg meat soup rice pie jam honey
candy cookie donut muffin pancake waffle cracker pretzel popcorn nut seed
horse cow pig sheep goat hen rooster chick goose swan owl crow robin sparrow
mouse rat rabbit squirrel fox wolf deer moose lion tiger zebra monkey panda koala
elephant giraffe camel donkey pony puppy kitten lamb calf foal piglet
snake lizard turtle frog toad crab clam snail worm ant bee wasp fly moth bug beetle
spider butterfly ladybug dragonfly grasshopper cricket
tree bush flower rose tulip daisy lily fern grass weed vine moss seed root branch
leaf petal thorn bark log stick stone rock pebble sand mud clay dirt dust
star moon sun cloud rain snow wind storm fog mist dew frost ice hail
hill cliff cave river lake pond stream pool wave beach shore island
hammer nail saw drill bolt screw wrench rake hoe shovel ladder fence gate post
candle torch flame match coal ash smoke
shirt pant dress coat cap glove scarf boot belt button zipper pocket collar
bag sack purse basket bucket jar bottle can tin crate barrel chest trunk
pen pencil crayon marker chalk paper card note map letter stamp envelope
flute horn bell whistle harp banjo fiddle piano organ chime gong rattle
boat ship raft canoe kayak sail anchor oar paddle deck mast
coin gold silver gem pearl jewel bead chain locket bracelet button medal
""".split()

seen, pool = set(), []
for w in CANDIDATES:
    w = w.lower()
    if w in seen:
        continue
    seen.add(w)
    pieces = sp.encode("▁" + w if False else " " + w, out_type=str)  # leading-space piece
    rt = sp.id_to_piece(sp.piece_to_id("▁" + w)).replace("▁", "")
    if len(pieces) == 1 and rt == w:           # exactly one token AND round-trips
        pool.append(w)

rng = random.Random(20260628)
rng.shuffle(pool)
n_test = max(1, round(len(pool) * 0.20))
test = sorted(pool[:n_test])
train = sorted(pool[n_test:])

assert not (set(train) & set(test)), "OVERLAP!"
out = {"train": train, "test": test}
with open("data/wide_vocab.json", "w") as f:
    json.dump(out, f, indent=1)

print(f"single-token nouns in pool : {len(pool)}")
print(f"  train : {len(train)}")
print(f"  test  : {len(test)}  (held-out)")
print(f"  overlap: {len(set(train) & set(test))}  (must be 0)")
print(f"  sample train: {train[:12]}")
print(f"  sample test : {test[:12]}")
print("saved -> data/wide_vocab.json")
