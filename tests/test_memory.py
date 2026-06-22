import torch
from config import get_config
from core.memory import SolarRingMemory

cfg = get_config()


def test_memory():
    B, slots, d = 4, cfg.memory.slots, cfg.model.dim
    device = torch.device("cpu")

    srm = SolarRingMemory(cfg)

    # -------------------------
    # INIT
    # -------------------------
    m_t, m_s = srm.init_memory(B, device)

    assert m_t.shape == (B, slots, d)
    assert m_s.shape == (B, slots, d)
    assert not torch.isnan(m_t).any()

    # -------------------------
    # FORWARD
    # -------------------------
    h = torch.randn(B, d)

    h_out, m_t_new = srm(h, m_t)

    assert h_out.shape == (B, d)
    assert m_t_new.shape == (B, slots, d)
    assert not torch.isnan(h_out).any()
    assert not torch.isnan(m_t_new).any()

    # -------------------------
    # SPATIAL TEST
    # -------------------------
    m_s_above = torch.randn(B, slots, d)

    h_out2, m_t_new2 = srm(h, m_t, m_s_above=m_s_above)

    assert not torch.isnan(h_out2).any()

    # -------------------------
    # GRADIENT TEST (CORRECT)
    # -------------------------
    loss = h_out.mean()
    loss.backward()

    grads_exist = any(p.grad is not None for p in srm.parameters())
    assert grads_exist, "No gradients flowing"

    print("PASS ✅ test_memory — all checks passed")


if __name__ == "__main__":
    test_memory()