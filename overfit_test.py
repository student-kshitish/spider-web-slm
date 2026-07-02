"""
Overfit test: trains on ONE fixed batch for 300 steps with hard=True from step 0.

Runs in float32 (no autocast) to isolate Gumbel-STE gradient correctness from
the bf16 dtype question. A working architecture MUST drive CE toward near-zero
on a single repeated batch; if it plateaus around 8.5, the STE gradient path
is broken and full training won't help.
"""
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn as nn
import sentencepiece as spm

from config import get_config
from core.web import SpiderWeb
from train.dataloader import get_dataloader

torch.manual_seed(0)

cfg = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Float32 throughout — isolates gradient question from dtype question
model = SpiderWeb(cfg).to(device)

sp = spm.SentencePieceProcessor()
sp.Load("data/tokenizer.model")

loader = get_dataloader(cfg, sp)
x_fixed, y_fixed = next(iter(loader))
x_fixed = x_fixed.to(device)
y_fixed = y_fixed.to(device)

print(f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"Batch : {x_fixed.shape[0]} seqs × {x_fixed.shape[1]} tokens")
print(f"Mode  : float32, hard=True always, tau=0.1")
print(f"Steps : 300, log every 25\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
ce_fn = nn.CrossEntropyLoss()

model.train()
for step in range(300):
    out = model(x_fixed, tau=0.1, hard=True)
    logits = out["logits"].float()
    B, T, V = logits.shape
    loss = ce_fn(logits.view(-1, V), y_fixed.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 25 == 0:
        print(f"step {step:4d} | ce {loss.item():.4f}")

print("\nOverfit test complete.")
print("Expected: CE drops well below 8.5 toward ~0-2 by step 300.")
print("If CE stays near 8.5, STE gradient path (Gumbel hard=True) is broken.")
