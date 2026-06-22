import torch
from config import get_config
from core.web import SpiderWeb

cfg = get_config()
model = SpiderWeb(cfg)

path = "checkpoints/best.pt"
device = torch.device("cpu")

ckpt = torch.load(path, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state)

print("✅ Model loaded successfully")
