# decoding_compare.py — before/after comparison of decoding improvements

import torch
from contextlib import nullcontext
import sentencepiece as spm

from config import get_config
from core.web import SpiderWeb
from infer import generate, load_checkpoint, INFER_TAU

CKPT   = "checkpoints/final_run/best.pt"
SEQ    = 128
SEED   = 42
N_TOK  = 80   # generous budget; sentence-stop trims it naturally

torch.manual_seed(SEED)


def load():
    cfg = get_config()
    cfg.model.dim         = 64
    cfg.model.hidden_dim  = 256
    cfg.model.max_seq_len = SEQ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpiderWeb(cfg).to(device)
    ckpt   = torch.load(CKPT, map_location=device)
    state  = {k: v.float() if v.is_floating_point() else v
               for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")
    return model, sp, cfg, device


def run(model, sp, cfg, device, prompt, **kwargs):
    torch.manual_seed(SEED)
    return generate(
        model=model, sp=sp, prompt=prompt,
        max_new_tokens=N_TOK, seq_len=SEQ, device=device,
        **kwargs,
    )


def sep(title=""):
    if title:
        print(f"\n  ── {title}")
    else:
        print()


def block(label, text):
    print(f"  [{label}]")
    print(f"  {text}")


if __name__ == "__main__":
    model, sp, cfg, device = load()
    print(f"Loaded {CKPT}  dim=64\n")

    prompts = ["Once upon a time", "Lily wanted to"]

    # ── baseline settings (old generate() defaults) ───────────────────────
    OLD = dict(temperature=0.7, top_k=40, top_p=0.9,
               no_repeat_ngram_size=0, stop_at_sentence=False)

    # ── three candidate new settings ──────────────────────────────────────
    SETTINGS = {
        "A  temp=0.7 top_p=0.9 ngram=3 stop":
            dict(temperature=0.7, top_k=0,  top_p=0.9,
                 no_repeat_ngram_size=3, stop_at_sentence=True, min_new_tokens=15),
        "B  temp=0.6 top_k=30  ngram=3 stop":
            dict(temperature=0.6, top_k=30, top_p=1.0,
                 no_repeat_ngram_size=3, stop_at_sentence=True, min_new_tokens=15),
        "C  temp=0.8 top_p=0.92 ngram=3 stop":
            dict(temperature=0.8, top_k=0,  top_p=0.92,
                 no_repeat_ngram_size=3, stop_at_sentence=True, min_new_tokens=15),
    }

    # ── greedy + ngram (does it still collapse?) ──────────────────────────
    GREEDY_OLD  = dict(temperature=0, top_k=0, top_p=1.0,
                       no_repeat_ngram_size=0, stop_at_sentence=False)
    GREEDY_NGRAM = dict(temperature=0, top_k=0, top_p=1.0,
                        no_repeat_ngram_size=3, stop_at_sentence=True, min_new_tokens=10)

    for prompt in prompts:
        print("━" * 62)
        print(f'  PROMPT: "{prompt}"')
        print("━" * 62)

        sep("OLD (temp=0.7 top_k=40 top_p=0.9, no ngram block)")
        block("OLD", run(model, sp, cfg, device, prompt, **OLD))

        for label, kwargs in SETTINGS.items():
            sep(label)
            block("NEW", run(model, sp, cfg, device, prompt, **kwargs))

        sep("GREEDY — old (no ngram block)")
        block("GREEDY-OLD", run(model, sp, cfg, device, prompt, **GREEDY_OLD))

        sep("GREEDY — with ngram=3 + sentence-stop")
        block("GREEDY-NGRAM", run(model, sp, cfg, device, prompt, **GREEDY_NGRAM))

        print()

    # ── pick the best setting and show a clean final comparison ───────────
    print("━" * 62)
    print("  BEST SETTING SUMMARY  (Setting C: temp=0.8 top_p=0.92 ngram=3)")
    print("━" * 62)
    BEST = SETTINGS["C  temp=0.8 top_p=0.92 ngram=3 stop"]
    for prompt in prompts:
        print(f'\n  Prompt: "{prompt}"')
        block("BEFORE", run(model, sp, cfg, device, prompt, **OLD))
        block("AFTER ", run(model, sp, cfg, device, prompt, **BEST))
