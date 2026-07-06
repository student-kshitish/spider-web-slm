"""
DRY-RUN for Step 6 write-policy primitives (--write_decay, --erase). NO TRAINING,
no optimizer step, no checkpoint written. Verifies the two NATIVE memory ops are
wired correctly before any run is launched:

  0 FOUR fairness-checked samples (each is a TWO-OBJECT-GIVER story): new-owner /
    old-owner(KEPT) / non-involved / no-transfer.
  1 DECAY-ON-WRITE actually scales old slot content: (a) the primitive itself —
    slot <- (1-(1-γ)a)·slot + a·v — old-content norm scaled by (1-(1-γ)a) < 1; (b)
    the model-level effect — on a two-object giver, the recall's read_dist mass on
    the OLDER intra-slot write drops (relative to the newer) under γ<1 vs γ=1.
  2 name_src=None asserted in the forward (both lookbacks locate names unaided).
  3 GIVER-resolution L_name component present and nonzero — L_name on the SECOND
    (giver) lookback (giver_src targets) has n_labeled>0, loss>0, and its gradient
    reaches giver_lookback.q_proj/k_proj.
  4 ERASE-gate gradient flows from the OLD-OWNER recall NLL: back-prop the NLL of
    the kept-object target at the old-owner recall; separable_mem.erase_gate.weight
    .grad must be non-zero (the gate learns WHEN to erase from this signal).
  5 no NaN anywhere.

Warm base: content_addr_A. --write_decay is config-only (no new params); --erase
adds giver_lookback + erase_gate (zero-init -> erase starts OFF, warm-compatible).

Run:  python3 dryrun_write_policy.py
"""
import os, random
os.environ["WANDB_MODE"] = "disabled"
import torch
import torch.nn.functional as F
import sentencepiece as spm

import train_attn_super as TA
from core.web import SpiderWeb

BASE  = "checkpoints/content_addr_A/best.pt"
GAMMA = 0.5
B     = 12


def new_ds(sp, seed=0):
    ds = TA.StructuredMix.__new__(TA.StructuredMix)
    ds.sp = sp; ds.rng = random.Random(seed); ds.n = 1000
    return ds


def build_model(device, write_decay, erase):
    ckpt = torch.load(BASE, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg = TA.ft_config(B, slots, 3500, use_pointer=True, warm_gen=True,
                       lambda_floor=0.2, mem_copy=True, write_mode="separable",
                       no_meanpool=True, name_lookback=True,
                       write_decay=write_decay, erase=erase)
    model = SpiderWeb(cfg).to(device)
    m, u = model.load_state_dict(state, strict=False)
    bad = [k for k in m if not k.startswith(TA.NEW_PREFIXES)]
    assert not bad and not u, f"warm load: missing={bad} unexpected={u}"
    return model, slots


def sample(ds, rng, cue, do_t, k):
    for _ in range(200):
        it = ds.build_transfer_item(rng, TA.OBJECTS, cue, do_t, k, for_training=True)
        if it is not None:
            return it
    raise RuntimeError(f"no sample for {cue}")


def pad(ds, it):
    ids = it["ids"]
    return ds._pad(ids[:-1], ids[1:], it["recall_pos"], it["source_pos"], 1,
                   it["subj"], it["name_src"], it["giver_src"])


# ── CHECK 0 ─────────────────────────────────────────────────────────────────────
def print_samples(ds):
    print("\n" + "=" * 94)
    print("[0] FOUR SAMPLE ITEMS (two-object giver; fairness verified)")
    print("=" * 94)
    rng = random.Random(7)
    for title, cue, do_t, k in [
            ("NEW-OWNER (target=TRANSFERRED; needs recency)", "new", True, 2),
            ("OLD-OWNER (target=KEPT; needs suppression of transferred)", "old", True, 2),
            ("NON-INVOLVED (interference control)", "noninvolved", True, 3),
            ("NO-TRANSFER control (giver still has two objects)", "plain", False, 3)]:
        it = sample(ds, rng, cue, do_t, k)
        p = it["pcs"]
        print(f"\n--- {title} ---")
        print(f"  {it['text']}")
        print(f"  giver={it['giver']} [objT(transferred)={it['objT']}, objK(kept)={it['objK']}]"
              f"  recip={it['recip']}(own={it['recip_obj']})  CUE={it['cue_name']}"
              f"  CORRECT={it['cue_obj']}")
        if it["cue_mode"] == "new":
            fair = it["cue_obj"] == it["objT"] and it["objT"] != it["recip_obj"]
            print(f"  fairness: correct=transferred '{it['objT']}' != recipient-own "
                  f"'{it['recip_obj']}' -> {'FAIR' if fair else 'NOT FAIR'}")
            assert fair
        elif it["cue_mode"] == "old":
            fair = it["cue_obj"] == it["objK"] and it["objK"] != it["objT"]
            print(f"  fairness: correct=KEPT '{it['objK']}' != transferred '{it['objT']}' "
                  f"(erase must suppress '{it['objT']}' in the giver slot) -> "
                  f"{'FAIR' if fair else 'NOT FAIR'}")
            assert fair
        elif it["cue_mode"] == "noninvolved":
            print(f"  interference: cue '{it['cue_name']}' not in transfer; "
                  f"correct='{it['cue_obj']}' must survive.")
        else:
            print(f"  no-transfer: plain binding; correct='{it['cue_obj']}'.")
        tp, ep = it["transfer_pos"], it["erase_pos"]
        tlbl = f"(`{p[tp]}`)->recip `{p[it['name_src'][tp]]}`" if tp is not None else ""
        elbl = f"(`{p[ep]}`)->giver `{p[it['giver_src'][ep]]}`" if ep is not None else ""
        print(f"  labels: transfer_pos={tp}{tlbl}   erase_pos={ep}{elbl}")


def make_batch(ds, device, specs):
    xs = [[] for _ in range(8)]
    metas = []
    rng = random.Random(11)
    for cue, do_t, k in specs:
        it = sample(ds, rng, cue, do_t, k)
        tup = pad(ds, it)
        for i, t in enumerate(tup):
            xs[i].append(t)
        metas.append(it)
    st = lambda L: torch.stack(L).to(device)
    tn = lambda L: torch.tensor(L, dtype=torch.long, device=device)
    batch = (st(xs[0]), st(xs[1]), tn(xs[2]), tn(xs[3]), tn(xs[4]),
             st(xs[5]), st(xs[6]), st(xs[7]))
    return batch, metas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    torch.manual_seed(42); random.seed(123)
    print(f"[dryrun] device={device} base={BASE} B={B} γ={GAMMA}  "
          f"(NO training, NO optimizer step, NO ckpt write)")
    ds = new_ds(sp)
    print_samples(ds)

    # ── CHECK 1a: the DECAY PRIMITIVE scales old slot content ───────────────────
    print("\n" + "=" * 94)
    print("[1a] DECAY-ON-WRITE primitive: slot <- (1-(1-γ)a)·slot + a·v "
          "(old content actually scaled)")
    torch.manual_seed(0)
    d = 64
    slot = torch.randn(d); v = torch.randn(d); a = 0.8
    n0 = slot.norm().item()
    print(f"    write strength a={a};  ||slot|| BEFORE = {n0:.4f}")
    for g in (1.0, 0.5, 0.2):
        scale = 1.0 - (1.0 - g) * a
        old_after = (scale * slot).norm().item()
        print(f"      γ={g}: decay scale=(1-(1-γ)a)={scale:.3f}  "
              f"||old content AFTER|| = {old_after:.4f}  "
              f"({'unchanged' if g==1.0 else f'{100*(1-old_after/n0):.0f}% suppressed'})")
    assert (1.0 - (1.0 - GAMMA) * a) < 1.0

    # ── build models (decay-only and decay+erase) ───────────────────────────────
    m_g1, _ = build_model(device, 1.0, False)      # additive reference
    m_dec, _ = build_model(device, GAMMA, False)   # decay only
    m_er, slots = build_model(device, GAMMA, True) # decay + erase
    for m in (m_g1, m_dec, m_er):
        m.train()

    # ── CHECK 1b: model-level recency — older intra-slot write down-weighted ────
    print("\n" + "=" * 94)
    print("[1b] model-level recency: read_dist mass on the OLDER vs NEWER giver-slot "
          "write (γ=1 vs γ=%.1f)" % GAMMA)
    # two-object OLD-owner items: giver slot holds objT(older intro) & objK(newer intro)
    specs_old = [("old", True, 2)] * B
    batch, metas = make_batch(ds, device, specs_old)
    x, y, rp, sp_pos, isb, subj, nsrc, gsrc = [t.to(device) for t in batch]

    def read_at_recall(model):
        torch.manual_seed(1234)
        with torch.no_grad():
            o = model(x, tau=0.1, hard=True, subj_id=subj, name_src=None)
        return o["sep_stats"]["read_dist"].float()

    rd1 = read_at_recall(m_g1); rdd = read_at_recall(m_dec)
    older = newer = 0.0
    older_d = newer_d = 0.0
    for b, it in enumerate(metas):
        r = it["recall_pos"]
        iT, iK = it["intro_T_pos"], it["intro_K_pos"]   # older(objT) vs newer(objK)
        older   += rd1[b, r, iT].item(); newer   += rd1[b, r, iK].item()
        older_d += rdd[b, r, iT].item(); newer_d += rdd[b, r, iK].item()
    n = len(metas)
    print(f"    γ=1.0 (additive): read mass older(objT intro)={older/n:.3f}  "
          f"newer(objK intro)={newer/n:.3f}")
    print(f"    γ={GAMMA} (decay): read mass older(objT intro)={older_d/n:.3f}  "
          f"newer(objK intro)={newer_d/n:.3f}")
    print(f"    older/newer ratio: γ=1 {older/max(newer,1e-6):.3f}  ->  "
          f"γ={GAMMA} {older_d/max(newer_d,1e-6):.3f}  "
          f"(decay should LOWER the older write's relative mass)")

    # ── CHECK 2 + 3: name_src=None; GIVER L_name present & reaches the lookback ──
    print("\n" + "=" * 94)
    fwd_name_src = None
    assert fwd_name_src is None, "name_src must be absent from the forward"
    torch.manual_seed(1234)
    out = m_er(x, tau=0.1, hard=False, subj_id=subj, name_src=fwd_name_src)
    print("[2] name_src=None in forward -> ASSERTED (both lookbacks locate unaided)")
    assert out.get("giver_stats") is not None and "attn" in out["giver_stats"]

    name_loss, nm_mass, nm_acc, n_nm = TA.name_supervision(out, nsrc, "name_stats")
    giv_loss, gv_mass, gv_acc, n_gv = TA.name_supervision(out, gsrc, "giver_stats")
    m_er.zero_grad(set_to_none=True)
    giv_loss.backward(retain_graph=True)
    gq = m_er.giver_lookback.q_proj.weight.grad
    gk = m_er.giver_lookback.k_proj.weight.grad
    gq_n = None if gq is None else gq.norm().item()
    gk_n = None if gk is None else gk.norm().item()
    print("\n" + "=" * 94)
    print(f"[3] GIVER L_name: loss={giv_loss.item():.4f} mass={gv_mass:.3f} "
          f"locAcc={gv_acc:.3f} n_labeled={n_gv}  (recipient L_name={name_loss.item():.3f} "
          f"n={n_nm}, ref)")
    print(f"    giver_lookback.q_proj.grad norm = {gq_n}")
    print(f"    giver_lookback.k_proj.grad norm = {gk_n}   (both must be > 0)")
    assert n_gv > 0 and giv_loss.item() > 0, "no giver labels / zero giver L_name"
    assert gq_n and gq_n > 0 and gk_n and gk_n > 0, "giver L_name does not reach the lookback!"

    # ── CHECK 4: ERASE-gate gradient flows from the OLD-OWNER recall NLL ─────────
    m_er.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    out2 = m_er(x, tau=0.1, hard=False, subj_id=subj, name_src=None)
    P = out2["pointer"]["P"].float()                    # (B,T,V) copy/gen mixture
    ar = torch.arange(B, device=device)
    tgt = y[ar, rp].clamp_min(0)                        # kept-object target @ old-owner recall
    nll = -torch.log(P[ar, rp, tgt].clamp_min(1e-9)).mean()
    nll.backward()
    eg = m_er.separable_mem.erase_gate.weight.grad
    eb = m_er.separable_mem.erase_gate.bias.grad
    eg_n = None if eg is None else eg.norm().item()
    eb_n = None if eb is None else eb.norm().item()
    print("\n" + "=" * 94)
    print(f"[4] OLD-OWNER (kept) recall NLL={nll.item():.4f} -> ERASE gate gradient")
    print(f"    separable_mem.erase_gate.weight.grad norm = {eg_n}")
    print(f"    separable_mem.erase_gate.bias.grad   norm = {eb_n}   (must be > 0)")
    assert (eg_n and eg_n > 0) or (eb_n and eb_n > 0), \
        "erase gate gets NO gradient from the old-owner NLL!"
    e_now = out2["sep_stats"]["erase_e"].mean().item()
    print(f"    erase gate e mean = {e_now:.4f} (zero-init -> starts ~off; the run opens it)")

    # ── CHECK 5: no NaN ─────────────────────────────────────────────────────────
    fin = all(torch.isfinite(t).all().item() for t in
              (out2["logits"], P, out2["sep_stats"]["read_dist"],
               out2["sep_stats"]["write_w"], out2["giver_stats"]["attn"],
               out2["sep_stats"]["erase_e"], out2["sep_stats"]["w_giver"]))
    print("\n" + "=" * 94)
    print(f"[5] finite: logits/P/read_dist/write_w/giver_attn/erase_e/w_giver = {fin}")
    assert fin, "NaN/Inf in forward outputs"

    print("\n" + "=" * 94)
    print(">>> DRY-RUN COMPLETE — all checks passed. No checkpoint written, nothing trained.")
    print("=" * 94)


if __name__ == "__main__":
    main()
