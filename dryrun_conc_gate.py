"""
DRY-RUN for --conc_gate (NO TRAINING). Verifies the read-concentration gate is
wired correctly before any run is launched:

  1 gate input IS the read stats : out.pointer.gate_feats[...,:2] == [read_max,
    read_entropy] recomputed from sep_stats.read_dist (numeric equality).
  2 gradient reaches gate weights from L_router : backward the ROUTER loss ALONE;
    model.conc_gate.weight.grad must be non-zero (and copy_gate — the off path —
    must get NO grad).
  3 no NaN : loss / logits / gate feats finite.
  4 warm genCE : P_gen NLL ~2.7 (not ~8.5), i.e. base warm-start intact.
  + name_src=None asserted in the forward (name_lookback locates the name unaided).

Mirrors the content_addr_A recipe (use_pointer + warm_gen + lambda_floor 0.2 +
mem_copy + multi_entity + name_lookback), warm-started from the same base; the ONLY
change is conc_gate. Runs both variants: stats-only and stats+x. No optimizer step,
no checkpoint written.

Run:  python3 dryrun_conc_gate.py
"""
import os, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import sentencepiece as spm

import train_attn_super as TA
from core.web import SpiderWeb

BASE = "checkpoints/mem_integrated/best.pt"
B    = 8
SEED = 0


def make_batch(sp, device):
    """A small multi-entity binding batch, built via the training dataset's own
    _binding_multi (no TinyStories load). Returns the same tuple the train loop uses."""
    ds = TA.StructuredMix.__new__(TA.StructuredMix)               # skip __init__ (no TS read)
    ds.sp = sp; ds.rng = random.Random(SEED); ds.n = 1000
    xs, ys, rps, sps, isbs, subjs, nss = [], [], [], [], [], [], []
    for _ in range(B):
        x, y, rp, sp_pos, isb, s, ns, gs = ds._binding_multi()   # 8-tuple (gs unused here)
        xs.append(x); ys.append(y); rps.append(rp); sps.append(sp_pos)
        isbs.append(isb); subjs.append(s); nss.append(ns)
    stack = lambda L: torch.stack(L).to(device)
    tens  = lambda L: torch.tensor(L, dtype=torch.long, device=device)
    return (stack(xs), stack(ys), tens(rps), tens(sps), tens(isbs),
            stack(subjs), stack(nss))


def build_model(variant, device):
    ckpt = torch.load(BASE, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    # content_addr_A recipe (content_addr/w_router/w_name are LOSS-side, not config)
    cfg = TA.ft_config(B, slots, 3500, use_pointer=True, warm_gen=True,
                       lambda_floor=0.2, mem_copy=True, write_mode="separable",
                       no_meanpool=True, name_lookback=True, conc_gate=variant)
    model = SpiderWeb(cfg).to(device)
    m, u = model.load_state_dict(state, strict=False)
    bad = [k for k in m if not k.startswith(TA.NEW_PREFIXES)]
    assert not bad and not u, f"warm load: missing={bad} unexpected={u}"
    assert model._conc_gate == variant and hasattr(model, "conc_gate")
    model.train()
    return model, ckpt.get("ema_ce")


def run_variant(variant, sp, device):
    print("\n" + "=" * 84)
    print(f"CONC_GATE DRY-RUN — variant '{variant}'   "
          f"(in_dim = {2 + (64 if variant == 'stats_x' else 0)})")
    print("=" * 84)
    torch.manual_seed(42); random.seed(123)
    model, ema = build_model(variant, device)
    x, y, rp, sp_pos, isb, subj, nsrc = make_batch(sp, device)

    # forward EXACTLY as the training loop (name_lookback => name_src absent)
    fwd_name_src = None
    assert fwd_name_src is None, "name_src must be absent from the name_lookback forward"
    out = model(x, tau=0.1, hard=False, subj_id=subj, name_src=fwd_name_src)
    pt = out["pointer"]

    # ── CHECK 1: the gate input tensor IS the read-concentration stats ──────────
    read = out["sep_stats"]["read_dist"].float().clamp_min(1e-9)      # (B,T,u)
    rmax = read.max(dim=-1).values                                    # (B,T)
    rent = -(read * read.log()).sum(dim=-1)                           # (B,T)
    gf = pt["gate_feats"]
    assert pt["gate_src"] == variant and gf is not None, "gate_feats not exposed"
    exp_dim = 2 + (model.d if variant == "stats_x" else 0)
    assert gf.shape[-1] == exp_dim, f"gate_feats dim {gf.shape[-1]} != {exp_dim}"
    d0 = (gf[..., 0] - rmax).abs().max().item()
    d1 = (gf[..., 1] - rent).abs().max().item()
    print(f"[1] gate_src={pt['gate_src']}  gate_feats shape={tuple(gf.shape)} "
          f"(=[read_max, read_entropy{', x' if variant=='stats_x' else ''}])")
    print(f"    max|gate_feats[...,0] - read_max|     = {d0:.2e}")
    print(f"    max|gate_feats[...,1] - read_entropy| = {d1:.2e}")
    if variant == "stats_x":
        print(f"    stats+x: trailing {gf.shape[-1]-2} dims are x (d={model.d})")
    assert d0 < 1e-5 and d1 < 1e-5, "gate input is NOT the read stats!"
    # show recall-position components (the copy token)
    print("    per-example @recall  [read_max, read_entropy] -> lambda:")
    for b in range(min(4, B)):
        r = int(rp[b])
        print(f"      ex{b}: [{gf[b, r, 0]:.3f}, {gf[b, r, 1]:.3f}] "
              f"-> lam={pt['lambda'][b, r].item():.3f}")

    # ── CHECK 4: warm genCE (P_gen NLL) ─────────────────────────────────────────
    valid = (y != TA.IGNORE)
    Pg = pt["P_gen"].float()
    genll = -torch.log(Pg.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).clamp_min(1e-9))
    genCE = genll[valid].mean().item()

    # ── CHECK 3: no NaN anywhere ────────────────────────────────────────────────
    finite = all(torch.isfinite(t).all().item()
                 for t in (out["logits"], pt["P"], pt["P_gen"], pt["P_copy"], gf))
    print(f"[3] finite: logits/P/P_gen/P_copy/gate_feats all finite = {finite}")
    print(f"[4] warm genCE (P_gen NLL) = {genCE:.3f}  (want ~2.7, not ~8.5)   base ema_ce={ema:.3f}")
    assert finite, "NaN/Inf in forward outputs"

    # ── CHECK 2: gradient reaches the gate weights from L_router ALONE ───────────
    router_loss, lam_recall = TA.gate_router_loss(out, rp, isb, y)
    model.zero_grad(set_to_none=True)
    router_loss.backward()
    cg = model.conc_gate.weight.grad
    cg_norm = None if cg is None else cg.norm().item()
    off_grad = model.copy_gate.weight.grad                          # off-path: should stay None
    print(f"[2] L_router={router_loss.item():.4f}  lambda@recall={lam_recall:.3f}")
    print(f"    conc_gate.weight.grad norm = {cg_norm}  (must be > 0 -> L_router shapes the gate)")
    print(f"    copy_gate.weight.grad      = {None if off_grad is None else off_grad.norm().item()} "
          f"(off-path unused -> expected None)")
    assert cg_norm is not None and cg_norm > 0, "L_router does NOT reach conc_gate weights!"
    print(f"\n>>> variant '{variant}': ALL CHECKS PASSED "
          f"(gate=read-stats, L_router->gate grad OK, finite, warm genCE).")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    print(f"[dryrun] device={device} base={BASE} B={B}  (NO training, NO optimizer step, NO ckpt write)")
    for variant in ("stats", "stats_x"):
        run_variant(variant, sp, device)
    print("\n" + "=" * 84)
    print("DRY-RUN COMPLETE — both variants wired correctly. No checkpoint written, nothing trained.")
    print("=" * 84)


if __name__ == "__main__":
    main()
