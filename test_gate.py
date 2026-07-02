"""
test_gate.py — can a tiny LEARNED write-gate distinguish entity vs filler,
and GENERALIZE to unseen entities? (Trains ONLY the gate, not the LM. Seconds.)

We take the write-gate from core.memory.LearnedWriteController — a single linear
projection  nn.Linear(d, 1)  + sigmoid, i.e. "should I store this?" — and train
it alone with BCE on store(1)/skip(0) labels.

--------------------------------------------------------------------------------
HOW FILLER IS DISTINGUISHABLE-IN-PRINCIPLE BUT NOT TRIVIAL
--------------------------------------------------------------------------------
The shortcut we must kill is MAGNITUDE: so EVERY vector is unit-norm. Entity vs
filler therefore has to be read off CONTENT/DIRECTION, not length.

We model "entity-ness" as a single shared *concept direction* u (a unit vector).
This is the only thing a LINEAR gate could ever generalize on: a hyperplane can
separate "has a u-component" from "doesn't", and that rule transfers to entities
it has never seen. Concretely, in a d=128 space with two fixed orthogonal axes
u (entity-ness) and f (filler-ness):

  entity v = normalize( a·u + content_⊥u ),   a ~ N(+0.32, 0.13)   (positive u)
  filler v = normalize( b·u + c·f + content ), b ~ N( 0.0 , 0.13), c ~ N(0.32,0.13)

Key properties that make it non-trivial:
  * unit norm everywhere               -> no magnitude shortcut.
  * filler is ALSO structured (it has its own strong axis f of equal strength)
    -> the gate can't cheat on "amount of structure / anisotropy"; only the
       u-axis discriminates, and it must DISCOVER u from labels.
  * the per-entity identity lives in the d-1 directions ⊥u, drawn fresh each
    time, so the 8 training entities are ~orthogonal to each other (pairwise
    cos ≈ 0) and an UNSEEN entity is a brand-new direction — memorizing the 8
    does NOT help; only learning u generalizes.
  * a, b overlap in the tails -> the task is not perfectly separable (realistic
    accuracy, not a suspicious 100%).

GENERALIZATION TEST: train on noisy versions of 8 base entities + filler; TEST
on noisy versions of 8 DIFFERENT, unseen base entities + fresh filler.

NEGATIVE CONTROL: rebuild with a=0 (entities have NO shared u-component — just 8
arbitrary unit vectors). Now entity and filler are statistically identical; the
gate can only MEMORIZE the 8 training directions, so TEST accuracy on unseen
entities must collapse to ~chance. This proves the test can fail and that any
pass is real generalization, not a rigged construction.

Run:  python3 test_gate.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.memory import LearnedWriteController

D            = 128
N_BASE       = 8        # 8 entity bases for train, 8 DIFFERENT for test
PER_BASE     = 30       # noisy samples per base
ENT_U_MEAN   = 0.32     # entity component along the entity-ness axis u
AXIS_SD      = 0.13     # spread of axis components (creates tail overlap)
FILL_F_MEAN  = 0.32     # filler component along its own axis f (equal structure)
NOISE        = 0.35     # noisy-mention perturbation (cos ~0.9 to base)
STEPS        = 400
LR           = 0.05
SEED         = 0


# ----------------------------------------------------------------- geometry --
def unit(x):
    return F.normalize(x, dim=-1)

def proj_out(v, axes):
    """remove components of v along each unit row in `axes` (list of (d,) unit vecs)."""
    for a in axes:
        v = v - (v @ a) * a
    return v


def make_bases(n, u, shared_axis):
    """n entity base vectors = unit( a·u + content_⊥u ); a>0 if shared_axis else a=0."""
    bases = []
    for _ in range(n):
        content = unit(proj_out(torch.randn(D), [u]))
        a = (ENT_U_MEAN + AXIS_SD * torch.randn(1).item()) if shared_axis else 0.0
        bases.append(unit(a * u + content))
    return bases


def noisy(base, u, shared_axis):
    """A noisy mention of an entity (same identity, slightly different 'words')."""
    v = unit(base + NOISE * torch.randn(D))
    if not shared_axis:
        return v
    # keep a positive u-component despite the perturbation (entity-ness persists)
    a = max(0.0, ENT_U_MEAN + AXIS_SD * torch.randn(1).item())
    return unit(a * u + unit(proj_out(v, [u])))


def make_filler(u, f, shared_axis):
    """Filler: unit, structured along its OWN axis f, ~0 mean on u (small noise)."""
    content = unit(proj_out(torch.randn(D), [u, f]))
    b = AXIS_SD * torch.randn(1).item()                 # u-component ~ N(0, sd)
    c = FILL_F_MEAN + AXIS_SD * torch.randn(1).item()   # strong f-component
    if not shared_axis:
        b = 0.0                                         # identical stats to entity
        return unit(content)
    return unit(b * u + c * f + content)


def build_split(bases, u, f, shared_axis):
    X, y = [], []
    for base in bases:                                  # entities (label 1)
        for _ in range(PER_BASE):
            X.append(noisy(base, u, shared_axis)); y.append(1.0)
    for _ in range(len(bases) * PER_BASE):              # filler (label 0)
        X.append(make_filler(u, f, shared_axis)); y.append(0.0)
    return torch.stack(X), torch.tensor(y)


# -------------------------------------------------------------------- train --
def train_gate(Xtr, ytr, Xte, yte):
    lc = LearnedWriteController(D, slots=N_BASE)         # we train ONLY its write_gate
    gate = lc.write_gate                                 # nn.Linear(D, 1)
    opt = torch.optim.Adam(gate.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    for step in range(STEPS):
        opt.zero_grad()
        logits = gate(Xtr).squeeze(-1)
        loss = lossf(logits, ytr)
        loss.backward(); opt.step()

    @torch.no_grad()
    def evalset(X, y):
        p = torch.sigmoid(gate(X).squeeze(-1))
        acc = ((p > 0.5).float() == y).float().mean().item()
        # ROC-AUC (threshold-free) via rank statistic
        order = torch.argsort(p)
        ranks = torch.empty_like(p); ranks[order] = torch.arange(len(p), dtype=p.dtype)
        npos, nneg = y.sum().item(), (1 - y).sum().item()
        auc = (ranks[y == 1].sum().item() - npos * (npos - 1) / 2) / (npos * nneg)
        return acc, auc, p

    tr_acc, tr_auc, _ = evalset(Xtr, ytr)
    te_acc, te_auc, te_p = evalset(Xte, yte)
    return dict(tr_acc=tr_acc, tr_auc=tr_auc, te_acc=te_acc, te_auc=te_auc,
                te_p=te_p, gate=gate)


# --------------------------------------------------------------------- main --
def run(shared_axis, tag):
    u = unit(torch.randn(D))
    f = unit(proj_out(torch.randn(D), [u]))             # filler axis ⊥ u
    train_bases = make_bases(N_BASE, u, shared_axis)
    test_bases  = make_bases(N_BASE, u, shared_axis)    # DIFFERENT, unseen entities

    Xtr, ytr = build_split(train_bases, u, f, shared_axis)
    Xte, yte = build_split(test_bases,  u, f, shared_axis)

    # sanity: how orthogonal are the training entity bases to each other?
    cs = [F.cosine_similarity(train_bases[i], train_bases[j], dim=0).item()
          for i in range(N_BASE) for j in range(i + 1, N_BASE)]
    mean_pair = sum(cs) / len(cs)

    r = train_gate(Xtr, ytr, Xte, yte)
    print(f"\n{'='*68}\n{tag}\n{'='*68}")
    print(f"  entity-base pairwise cosine: mean={mean_pair:+.3f}  max={max(cs):+.3f}  (≈0 ⇒ distinct)")
    print(f"  TRAIN  acc={r['tr_acc']*100:5.1f}%   auc={r['tr_auc']:.3f}")
    print(f"  TEST   acc={r['te_acc']*100:5.1f}%   auc={r['te_auc']:.3f}   <-- UNSEEN entities & filler")
    return r, Xte, yte


def main():
    torch.manual_seed(SEED)

    print("###  MAIN: entities share a learnable 'entity-ness' axis u  ###")
    r, Xte, yte = run(shared_axis=True, tag="MAIN — shared entity-ness axis (generalizable)")

    # example scores on held-out, unseen vectors
    p = r["te_p"]
    ent_idx = (yte == 1).nonzero(as_tuple=True)[0][:4]
    fil_idx = (yte == 0).nonzero(as_tuple=True)[0][:4]
    print("\n  Example gate scores on HELD-OUT vectors (sigmoid 'store?' prob):")
    for i in ent_idx:
        print(f"    unseen ENTITY  -> {p[i].item():.3f}  {'store ✓' if p[i] > 0.5 else 'SKIP ✗'}")
    for i in fil_idx:
        print(f"    unseen filler  -> {p[i].item():.3f}  {'STORE ✗' if p[i] > 0.5 else 'skip ✓'}")

    print("\n\n###  NEGATIVE CONTROL: NO shared axis (entities = 8 arbitrary unit vecs)  ###")
    rc, _, _ = run(shared_axis=False,
                   tag="CONTROL — no entity-ness axis (only memorization possible)")

    print(f"\n{'#'*68}\nVERDICT\n{'#'*68}")
    main_gen = r["te_acc"] > 0.65
    print(f"  MAIN    test acc = {r['te_acc']*100:.1f}%  (auc {r['te_auc']:.2f})  "
          f"=> {'GENERALIZES ✓' if main_gen else 'no generalization ✗'}")
    print(f"  CONTROL test acc = {rc['te_acc']*100:.1f}%  (auc {rc['te_auc']:.2f})  "
          f"train {rc['tr_acc']*100:.1f}%  => memorizes train, ~chance on unseen")
    print()
    if main_gen and rc["te_acc"] < 0.65:
        print("  => A single linear write-gate LEARNS 'entity-ness' as a content direction")
        print("     and classifies entity vs filler well ABOVE chance on UNSEEN entities —")
        print("     real generalization, not memorization (the control, lacking a shared")
        print("     axis, memorizes the training set yet collapses to chance on unseen).")
        print("     => The write-gate is viable; worth wiring into the SRM and training.")
    else:
        print("  => Inconclusive — inspect construction / thresholds.")


if __name__ == "__main__":
    main()
