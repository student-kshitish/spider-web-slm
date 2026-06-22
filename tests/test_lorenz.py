import torch
from config import get_config
from core.lorenz import LorenzRouter

cfg = get_config()

def test_lorenz():
    B, d = 4, cfg.model.dim
    router = LorenzRouter(cfg)
    M_t = torch.zeros(B, cfg.memory.slots, d)
    h   = torch.randn(B, d)

    # basic forward
    routing, h_guarded = router(h, M_t, tau=1.0, hard=False)
    assert routing.shape == (B, 3),    f"routing shape wrong: {routing.shape}"
    assert h_guarded.shape == (B, d),  f"h_guarded shape wrong: {h_guarded.shape}"
    assert not torch.isnan(routing).any(),   "routing has NaN"
    assert not torch.isnan(h_guarded).any(), "h_guarded has NaN"

    # routing sums to ~1 (softmax)
    sums = routing.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-4), \
        f"routing does not sum to 1: {sums}"

    # Lorenz clamp — xyz must stay bounded
    xyz = torch.tensor([[1000.0, 1000.0, 1000.0]])
    for _ in range(100):
        xyz = router._lorenz_step(xyz)
    assert xyz.abs().max().item() <= cfg.lorenz.clamp, \
        f"Lorenz clamp failed: max={xyz.abs().max():.1f}"

    # sensitivity — different inputs must give different routing
    h2 = h.flip(0)
    r2, _ = router(h2, M_t, tau=1.0)
    diff = (routing - r2).abs().mean().item()
    assert diff > 1e-4, f"Router not input-sensitive (diff={diff:.6f})"

    # hard routing — must be one-hot
    r_hard, _ = router(h, M_t, tau=0.1, hard=True)
    assert r_hard.max(dim=-1).values.min().item() == 1.0, \
        "Hard routing is not one-hot"

    print("PASS  test_lorenz — all checks passed")

if __name__ == "__main__":
    test_lorenz()