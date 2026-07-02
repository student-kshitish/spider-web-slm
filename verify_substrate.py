"""
Verify the three substrate fixes are wired correctly and isolated by flag.
  (1) undetach_mem    -> read->write gradient path is restored
  (2) residual_stream -> hop highway is additive (x = x + node_out)
  (3) sharp_head      -> spectral_norm dropped on FFN/aux/lm_head, KEPT on router/memory
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized

from config import Config, ModelConfig, MemoryConfig, LorenzConfig, RoutingConfig, TrainConfig
from core.web import SpiderWeb


def cfg(**flags):
    m = ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                    vocab_size=5000, max_seq_len=16, use_hybrid=True, lookback_width=-1,
                    **flags)
    return Config(model=m, memory=MemoryConfig(slots=32), lorenz=LorenzConfig(),
                  routing=RoutingConfig(max_hops=6),
                  train=TrainConfig(batch_size=2))


def is_spectral(mod):
    return is_parametrized(mod, "weight") or hasattr(mod, "weight_orig")


print("=" * 72)
print("(3) sharp_head — which projections kept / dropped spectral_norm")
print("=" * 72)
base = SpiderWeb(cfg(sharp_head=False))
sharp = SpiderWeb(cfg(sharp_head=True))
n0_b, n0_s = base.rings[0][0], sharp.rings[0][0]
checks = [
    ("node.w1 (FFN)",        n0_b.w1,            n0_s.w1,            "DROP"),
    ("node.w2 (FFN)",        n0_b.w2,            n0_s.w2,            "DROP"),
    ("node.w3 (FFN)",        n0_b.w3,            n0_s.w3,            "DROP"),
    ("node.aux_proj",        n0_b.aux_proj,      n0_s.aux_proj,      "DROP"),
    ("lm_head",              base.lm_head,       sharp.lm_head,      "DROP"),
    ("router.proj_in",       n0_b.router.proj_in,  n0_s.router.proj_in,  "KEEP"),
    ("router.proj_out",      n0_b.router.proj_out, n0_s.router.proj_out, "KEEP"),
    ("memory.q_proj",        n0_b.memory.q_proj,   n0_s.memory.q_proj,   "KEEP"),
    ("memory.k_proj",        n0_b.memory.k_proj,   n0_s.memory.k_proj,   "KEEP"),
    ("memory.gate_proj",     n0_b.memory.gate_proj, n0_s.memory.gate_proj, "KEEP"),
]
ok = True
for name, mb, ms, intent in checks:
    base_sn, sharp_sn = is_spectral(mb), is_spectral(ms)
    if intent == "DROP":
        good = base_sn and not sharp_sn
    else:
        good = base_sn and sharp_sn
    ok &= good
    print(f"  {name:<22} base_sn={base_sn!s:<5} sharp_sn={sharp_sn!s:<5} "
          f"intent={intent:<4} {'OK' if good else 'FAIL'}")
print(f"  => sharp_head isolation: {'OK' if ok else 'FAIL'}")

print("\n" + "=" * 72)
print("(2) residual_stream — additive hop highway changes the residual norm")
print("=" * 72)
torch.manual_seed(0)
x = torch.randint(0, 5000, (2, 16))
m_rep = SpiderWeb(cfg(residual_stream=False)); m_rep.eval()
m_add = SpiderWeb(cfg(residual_stream=True));  m_add.eval()
m_add.load_state_dict(m_rep.state_dict())  # identical weights, differ only by flag
with torch.no_grad():
    # capture residual entering final_norm via pre-hook
    caps = {}
    h1 = m_rep.final_norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("rep", i[0]))
    h2 = m_add.final_norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("add", i[0]))
    torch.manual_seed(1); m_rep(x, tau=0.1, hard=True)
    torch.manual_seed(1); m_add(x, tau=0.1, hard=True)
    h1.remove(); h2.remove()
print(f"  pre-final_norm residual norm:  replace={caps['rep'].norm():.2f}  "
      f"additive={caps['add'].norm():.2f}")
print(f"  => additive highway active: {'OK' if caps['add'].norm() > caps['rep'].norm()*1.2 else 'CHECK'}")

print("\n" + "=" * 72)
print("(1) undetach_mem — read->write gradient path restored")
print("=" * 72)
def mem_grad_norm(undetach):
    torch.manual_seed(0)
    m = SpiderWeb(cfg(undetach_mem=undetach)); m.train()
    xi = torch.randint(0, 5000, (2, 16))
    yi = torch.randint(0, 5000, (2, 16))
    out = m(xi, tau=0.5, hard=False)
    loss = F.cross_entropy(out["logits"].reshape(-1, 5000), yi.reshape(-1))
    m.zero_grad(); loss.backward()
    g = 0.0
    for n, p in m.named_parameters():
        if ".memory." in n and p.grad is not None:
            g += p.grad.float().norm().item() ** 2
    return g ** 0.5

g_det = mem_grad_norm(False)
g_undet = mem_grad_norm(True)
print(f"  memory-param grad norm  detached={g_det:.4f}   undetached={g_undet:.4f}")
print(f"  => extra read->write gradient path opened: "
      f"{'OK (grad changed)' if abs(g_undet - g_det) > 1e-6 else 'FAIL (no change)'}")

# the exact .detach() severance demo, for the record
w = torch.tensor([2.0], requires_grad=True); ((w * 3.0) * 5.0).sum().backward()
w2 = torch.tensor([2.0], requires_grad=True); written = (w2 * 3.0).detach(); (written * 5.0).sum()
print(f"\n  reminder: x.detach() severs grad (proved earlier: detached grad=None, "
      f"on-graph grad={w.grad.item()}). The fix uses the on-graph tensor.")
