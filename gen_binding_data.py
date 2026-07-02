"""
Step 1 — Build a small dependency-rich synthetic dataset.

Generates simple stories where a later token is predictable ONLY from an
earlier mention (cross-sentence binding), not from local context:

    "Lily had a red ball. She went to the park. She played on the swing.
     Then Lily threw the red ball."

Key property enforced:
  - the entity (NAME), the attribute (ATTR) and the object (OBJ) appear in
    the INTRO and the RECALL sentence only.
  - the filler sentences in between NEVER mention them, so the recalled word
    cannot be guessed from local context — it must be remembered.
  - distance between mention and recall is varied (1-3 filler sentences,
    ~4-20 tokens apart) by varying the number of fillers.

Saves to data/raw/binding.txt  (alongside, not replacing, TinyStories).
"""

import os
import random

random.seed(42)

TARGET_BYTES = 30 * 1024 * 1024   # ~30 MB
OUT_PATH     = "data/raw/binding.txt"

FEMALE = ["Lily", "Sara", "Mia", "Anna", "Lucy", "Emma", "Zoe", "Nina", "Ruby", "Ella"]
MALE   = ["Tom", "Ben", "Max", "Sam", "Leo", "Jack", "Tim", "Finn", "Jake", "Noah"]

OBJECTS = ["ball", "hat", "cat", "dog", "book", "cup", "kite", "doll", "box",
           "drum", "frog", "fish", "cake", "key", "shoe", "car", "bell",
           "duck", "bear", "boat"]

ATTRS = ["red", "blue", "green", "yellow", "big", "small", "soft", "old",
         "new", "shiny", "tiny", "fluffy", "round", "warm", "bright"]

INTRO = [
    "{name} had a {attr} {obj}.",
    "{name} found a {attr} {obj}.",
    "{name} got a {attr} {obj}.",
    "One day {name} saw a {attr} {obj}.",
]

# fillers MUST NOT mention obj / attr / another named entity
FILLER = [
    "{pron} went to the park.",
    "{pron} played in the yard.",
    "{pron} ran down the hill.",
    "{pron} sat by the tree.",
    "{pron} sang a happy song.",
    "{pron} walked to school.",
    "It was a sunny day.",
    "The sun was warm and bright.",
    "{pron} laughed and smiled.",
    "{pron} jumped up and down.",
    "{pron} took a little nap.",
    "Birds flew in the sky.",
]

# recall sentences depend on remembering name / attr / obj
RECALL = [
    "Then {name} {verb} the {attr} {obj}.",
    "{name} wanted the {attr} {obj} again.",
    "{Pron} picked up the {attr} {obj}.",
    "At last {name} found the {attr} {obj}.",
    "{Pron} loved the {attr} {obj} so much.",
    "Then {name} {verb} the {obj}.",
]

VERBS = ["threw", "held", "dropped", "kept", "hugged", "grabbed", "washed", "painted"]


def make_story():
    if random.random() < 0.5:
        name, pron, Pron = random.choice(FEMALE), "she", "She"
    else:
        name, pron, Pron = random.choice(MALE), "he", "He"

    attr = random.choice(ATTRS)
    obj  = random.choice(OBJECTS)
    verb = random.choice(VERBS)

    fields = dict(name=name, pron=pron, Pron=Pron, attr=attr, obj=obj, verb=verb)

    parts = [random.choice(INTRO).format(**fields)]

    n_filler = random.randint(1, 3)          # vary mention->recall distance
    for _ in range(n_filler):
        parts.append(random.choice(FILLER).format(**fields))

    parts.append(random.choice(RECALL).format(**fields))
    return " ".join(parts)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n = 0
    written = 0
    buf = []
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        while written < TARGET_BYTES:
            s = make_story()
            buf.append(s)
            n += 1
            if len(buf) >= 5000:
                chunk = "\n\n".join(buf) + "\n\n"
                f.write(chunk)
                written += len(chunk.encode("utf-8"))
                buf = []
        if buf:
            chunk = "\n\n".join(buf) + "\n\n"
            f.write(chunk)
            written += len(chunk.encode("utf-8"))

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote {OUT_PATH}")
    print(f"  stories : {n:,}")
    print(f"  size    : {size_mb:.1f} MB")
    print()
    print("=== 3 example stories ===")
    random.seed(7)
    for i in range(3):
        print(f"\n[{i+1}] {make_story()}")


if __name__ == "__main__":
    main()
