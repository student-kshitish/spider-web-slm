import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm

from config import get_config
from core.web import SpiderWeb


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


# ============================================================
# SAMPLING HELPERS
# ============================================================

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the k-th largest value."""
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out logits outside the top-p probability mass (nucleus sampling)."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # shift by one so the token that pushes cumsum above p is kept
    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > p
    sorted_logits[remove] = float("-inf")
    # scatter back to original order
    return logits.scatter(0, sorted_idx, sorted_logits)


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature
    else:
        # greedy
        return int(logits.argmax().item())

    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ============================================================
# GENERATION
# ============================================================

REPEAT_PENALTY = 1.3
REPEAT_WINDOW  = 32

# tau used at inference — matches the final annealed value from training
INFER_TAU = 0.1

# Sentence-ending token ids are resolved lazily at first call.
_SENT_END_IDS: list[int] | None = None

def _sent_end_ids(sp: spm.SentencePieceProcessor) -> list[int]:
    """Return token ids whose decoded surface form ends with . ! or ?"""
    global _SENT_END_IDS
    if _SENT_END_IDS is None:
        ends = set()
        for i in range(sp.GetPieceSize()):
            piece = sp.IdToPiece(i)
            if any(piece.endswith(c) for c in (".", "!", "?")):
                ends.add(i)
        _SENT_END_IDS = list(ends)
    return _SENT_END_IDS


def _block_ngrams(logits: torch.Tensor, generated: list[int], n: int) -> torch.Tensor:
    """Set logits to -inf for any token that would complete a repeated n-gram."""
    if n <= 0 or len(generated) < n - 1:
        return logits
    suffix = generated[-(n - 1):]          # last (n-1) tokens
    banned: set[int] = set()
    for start in range(len(generated) - (n - 1)):
        if generated[start : start + n - 1] == suffix:
            banned.add(generated[start + n - 1])
    if banned:
        logits = logits.clone()
        for tok in banned:
            logits[tok] = float("-inf")
    return logits


def generate(
    model: SpiderWeb,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seq_len: int,
    device: torch.device,
    # --- new parameters ---
    no_repeat_ngram_size: int = 0,
    stop_at_sentence: bool = False,
    min_new_tokens: int = 10,
) -> str:

    model.eval()

    prompt_ids = sp.EncodeAsIds(prompt)
    if not prompt_ids:
        prompt_ids = [sp.bos_id() if sp.bos_id() >= 0 else 1]

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []

    sent_ends = _sent_end_ids(sp) if stop_at_sentence else []

    with torch.no_grad():
        for _ in range(max_new_tokens):

            context = ids[:, -seq_len:] if ids.size(1) > seq_len else ids
            out     = model(context, tau=INFER_TAU, hard=True)
            logits  = out["logits"][0, -1, :]

            # repetition penalty over recent window
            recent_tokens = ids[0, -REPEAT_WINDOW:].tolist()
            for tok in set(recent_tokens):
                if logits[tok] > 0:
                    logits[tok] = logits[tok] / REPEAT_PENALTY
                else:
                    logits[tok] = logits[tok] * REPEAT_PENALTY

            # no-repeat-ngram blocking
            if no_repeat_ngram_size > 0:
                logits = _block_ngrams(logits, generated_ids, no_repeat_ngram_size)

            next_tok = sample_token(logits, temperature, top_k, top_p)

            generated_ids.append(next_tok)
            ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

            # sentence-boundary stop: halt after min_new_tokens at a sentence end
            if stop_at_sentence and len(generated_ids) >= min_new_tokens:
                if next_tok in sent_ends:
                    break

    return sp.DecodeIds(prompt_ids + generated_ids)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Spider Web SLM — Inference")
    parser.add_argument("--prompt",         type=str,   default="Once upon a time")
    parser.add_argument("--max_new_tokens",       type=int,   default=100)
    parser.add_argument("--temperature",          type=float, default=0.6)
    parser.add_argument("--top_k",                type=int,   default=30)
    parser.add_argument("--top_p",                type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int,   default=3)
    parser.add_argument("--stop_at_sentence",     action="store_true", default=False)
    parser.add_argument("--min_new_tokens",       type=int,   default=10)
    parser.add_argument("--checkpoint",           type=str,   default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = SpiderWeb(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------
    print(f"\nPrompt : {args.prompt}")
    print(f"Params : max_new_tokens={args.max_new_tokens}  temp={args.temperature}  "
          f"top_k={args.top_k}  top_p={args.top_p}\n")

    output = generate(
        model                = model,
        sp                   = sp,
        prompt               = args.prompt,
        max_new_tokens       = args.max_new_tokens,
        temperature          = args.temperature,
        top_k                = args.top_k,
        top_p                = args.top_p,
        seq_len              = cfg.model.max_seq_len,
        device               = device,
        no_repeat_ngram_size = args.no_repeat_ngram_size,
        stop_at_sentence     = args.stop_at_sentence,
        min_new_tokens       = args.min_new_tokens,
    )

    print("--- Generated ---")
    print(output)
    print("-----------------")


if __name__ == "__main__":
    main()
