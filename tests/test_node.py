import torch
from config import get_config
from core.node import WebNode

cfg = get_config()

def test_node():
    B, d, slots = 4, cfg.model.dim, cfg.memory.slots
    node = WebNode(cfg)
    srm  = node.memory

    m_t, _ = srm.init_memory(B, torch.device("cpu"))
    h       = torch.randn(B, d)
    nbrs    = torch.randn(B, 2, d)   # 2 lateral neighbours

    out = node(h, nbrs, m_t, m_s_above=None, tau=1.0, hard=False)

    assert "h"          in out, "missing key: h"
    assert "routing"    in out, "missing key: routing"
    assert "m_t"        in out, "missing key: m_t"
    assert "aux_logits" in out, "missing key: aux_logits"

    assert out["h"].shape          == (B, d),              f"h shape: {out['h'].shape}"
    assert out["routing"].shape    == (B, 3),              f"routing shape: {out['routing'].shape}"
    assert out["m_t"].shape        == (B, slots, d),       f"m_t shape: {out['m_t'].shape}"
    assert out["aux_logits"].shape == (B, cfg.model.vocab_size), \
        f"aux_logits shape: {out['aux_logits'].shape}"

    assert not torch.isnan(out["h"]).any(),          "h has NaN"
    assert not torch.isnan(out["routing"]).any(),    "routing has NaN"
    assert not torch.isnan(out["m_t"]).any(),        "m_t has NaN"
    assert not torch.isnan(out["aux_logits"]).any(), "aux_logits has NaN"

    # backward must not crash
    loss = out["h"].sum() + out["aux_logits"].sum()
    loss.backward()
    print("PASS  test_node — all checks passed")

if __name__ == "__main__":
    test_node()