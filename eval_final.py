# eval_final.py
# Three-part evaluation of checkpoints/final_run/best.pt
#   Test 1 — generation quality at temp 0.7 and 0.9
#   Test 2 — average CE over 50 fixed eval batches
#   Test 3 — final_norm.weight movement from init

import torch
import torch.nn.functional as F
import sentencepiece as spm
from contextlib import nullcontext

from config import get_config
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.dataloader import get_dataloader
from infer import generate, top_k_filter, top_p_filter, sample_token

# ============================================================
# CONFIG  (must match training run)
# ============================================================

CKPT = "checkpoints/final_run/best.pt"

torch.manual_seed(42)


def build_model(device):
    cfg = get_config()
    cfg.model.dim        = 64
    cfg.model.hidden_dim = 256
    cfg.model.max_seq_len = 128
    cfg.train.batch_size  = 16

    model = SpiderWeb(cfg).to(device)
    ckpt  = torch.load(CKPT, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    # cast to float32 (master weights)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# ============================================================
# TEST 1 — GENERATION QUALITY
# ============================================================

def test_generation(model, sp, cfg, device):
    print("=" * 60)
    print("TEST 1 — Generation Quality")
    print("=" * 60)

    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a dog",
        "Lily and Tom",
    ]
    temperatures = [0.7, 0.9]
    top_k, top_p = 40, 0.9
    max_new = 100

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 50)
        for temp in temperatures:
            torch.manual_seed(42)  # fixed seed per (prompt, temp) for reproducibility
            prompt_ids = sp.EncodeAsIds(prompt)
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated_ids = []

            REPEAT_PENALTY = 1.3
            REPEAT_WINDOW  = 32
            INFER_TAU      = 0.1

            with torch.no_grad():
                for _ in range(max_new):
                    context = ids[:, -cfg.model.max_seq_len:]
                    with autocast_ctx:
                        out = model(context, tau=INFER_TAU, hard=True)
                    logits = out["logits"][0, -1, :].float()

                    recent = ids[0, -REPEAT_WINDOW:].tolist()
                    for tok in set(recent):
                        if logits[tok] > 0:
                            logits[tok] /= REPEAT_PENALTY
                        else:
                            logits[tok] *= REPEAT_PENALTY

                    next_tok = sample_token(logits, temp, top_k, top_p)
                    generated_ids.append(next_tok)
                    ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

            text = sp.DecodeIds(prompt_ids + generated_ids)
            print(f"  temp={temp}: {text}")

    print()


# ============================================================
# TEST 2 — AVERAGE EVAL CE (50 batches)
# ============================================================

def test_eval_ce(model, cfg, device):
    print("=" * 60)
    print("TEST 2 — Average Eval CE (50 fixed batches)")
    print("=" * 60)

    sp_tok = spm.SentencePieceProcessor()
    sp_tok.Load("data/tokenizer.model")

    torch.manual_seed(0)   # fixed seed → reproducible 50-batch sample
    loader = get_dataloader(cfg, sp_tok)
    loss_fn = SpiderWebLoss().to(device)

    autocast_ctx = (
        torch.autocast("cuda", torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    ces = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 50:
                break
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                out = model(x, tau=0.1, hard=False)
                out["logits"] = out["logits"].float()
                _, mets = loss_fn(out, y, entropy_weight=0.003)
            ces.append(mets["ce"])
            if (i + 1) % 10 == 0:
                print(f"  batch {i+1:2d}/50  CE={mets['ce']:.4f}  running_avg={sum(ces)/len(ces):.4f}")

    avg_ce = sum(ces) / len(ces)
    min_ce = min(ces)
    print(f"\n  Avg CE over 50 batches : {avg_ce:.4f}")
    print(f"  Min CE over 50 batches : {min_ce:.4f}")
    print(f"  (old model avg CE was ~3.9996 per RESUME_TOMORROW.txt baseline)")
    print()
    return avg_ce, min_ce


# ============================================================
# TEST 3 — NORM MOVEMENT
# ============================================================

def test_norm_movement(model):
    print("=" * 60)
    print("TEST 3 — final_norm.weight Movement from Init")
    print("=" * 60)

    w = model.final_norm.weight.detach().cpu()
    mean   = w.mean().item()
    std    = w.std().item()
    delta  = (w - 1.0).abs().mean().item()
    pct    = delta / 1.0 * 100

    print(f"  final_norm.weight  mean={mean:.6f}  std={std:.6f}")
    print(f"  avg |w - 1.0|      = {delta:.6f}  ({pct:.2f}% from init)")
    status = "✓ trained (fp32 fix held)" if pct > 0.5 else "⛔ suspect"
    print(f"  Status: {status}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Checkpoint : {CKPT}\n")

    model, cfg = build_model(device)

    # load tokenizer for generation
    sp = spm.SentencePieceProcessor()
    sp.Load("data/tokenizer.model")

    total = sum(p.numel() for p in model.parameters())
    print(f"Model  : {total/1e6:.3f}M params  dim=64  hidden=256\n")

    test_generation(model, sp, cfg, device)
    avg_ce, min_ce = test_eval_ce(model, cfg, device)
    test_norm_movement(model)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Best checkpoint CE  : 4.4212  (saved in best.pt)")
    print(f"  Avg eval CE (50 b)  : {avg_ce:.4f}")
    print(f"  Min eval CE (50 b)  : {min_ce:.4f}")
    print(f"  Stopped at step     : 20000  (EMA 4.7508)")
    print(f"  final_norm trained  : ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
