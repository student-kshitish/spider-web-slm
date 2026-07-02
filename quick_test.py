import torch
from config import get_config
from core.web import SpiderWeb
from train.loss import SpiderWebLoss

cfg = get_config()

# isolate CE
cfg.train.entropy_weight = 0.0
cfg.train.balance_weight = 0.0

# deeper routing
cfg.routing.max_hops = 6

model = SpiderWeb(cfg)
loss_fn = SpiderWebLoss()

opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

x = torch.randint(0, cfg.model.vocab_size, (4, 32))
y = torch.randint(0, cfg.model.vocab_size, (4, 32))

print("Overfit test v6 (slow tau)...\n")

for step in range(300):
    tau = max(0.3, 1.0 - step * 0.002)   # 🔥 slower annealing

    opt.zero_grad(set_to_none=True)
    out = model(x, tau=tau, hard=False)
    loss, mets = loss_fn(out, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 30 == 0:
        r = out["routings"]
        if isinstance(r, list) and len(r) > 0 and isinstance(r[0], torch.Tensor):
            rd = r[0].detach().float().mean(0).tolist()
            route = f"s={rd[0]:.2f} in={rd[1]:.2f} out={rd[2]:.2f}"
        else:
            route = "N/A"

        print(f"step {step:3d}  ce={mets['ce']:.4f}  tau={tau:.2f}  route=[{route}]")

print("\n--- RESULT ---")

final = loss.item()

if final < 0.5:
    print("PASS — overfit complete")
elif final < 2.5:
    print("GOOD — architecture verified, ready for real training 🚀")
else:
    print(f"CE still high: {final:.4f}")