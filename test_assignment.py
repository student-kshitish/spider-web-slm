"""
test_assignment.py — ASSIGNMENT-controller diagnostic (no full training, minutes).

The storage test (test_separable_write.py) proved that, given PRE-SEGMENTED
entities, a non-blending write keeps them recoverable. This test attacks the
unsolved part: from a REALISTIC, messy stream — entities that repeat, interleave,
recur with noise, and are buried in ~50% non-entity "filler" — can a simple
content-similarity write-controller:

  1. allocate exactly ONE slot per distinct entity (not 5 slots for 5 mentions
     of "Lily", not 1 slot collapsing all entities),
  2. route NOISY repeats back to the SAME slot (merge, not split),
  3. avoid letting filler corrupt entity slots,

and afterwards still recover all 5 entities DISTINCTLY?

We run the heuristic controller in TWO configs:
  * GATED   : a salience write-gate (stand-in for a learned entity-detector)
              rejects filler  -> the realistic case.
  * UNGATED : the pure content heuristic exactly as specified (novel -> new
              slot), which CANNOT tell new-entity from new-filler, so filler
              also allocates slots. We check the fallback criterion: are the 5
              entity slots still cleanly recoverable despite the filler slots?

No language, no full training. Run:  python3 test_assignment.py
"""

import torch
import torch.nn.functional as F

from core.memory import HeuristicWriteController, LearnedWriteController

D      = 64
SEED   = 0
NAMES  = ["ball", "Lily", "key", "dog", "hat"]
N_ENT  = len(NAMES)
MENTIONS_PER_ENTITY = 3      # 1 first mention + 2 noisy repeats
NOISE_FRAC = 0.5             # noisy-repeat perturbation (-> cos ~0.89 to clean)
REPEAT_THRESH = 0.5
TEMP   = 0.05                # recovery-read softmax temperature
SLOTS  = 32                  # generous: enough for entities + filler in ungated run


# ------------------------------------------------------------------ data ----
def make_entities():
    """5 ~orthogonal unit entity vectors (QR of a random matrix)."""
    g = torch.randn(N_ENT, D)
    q, _ = torch.linalg.qr(g.t())          # (D, N_ENT) orthonormal columns
    return [q[:, i].contiguous() for i in range(N_ENT)]


def noisy(vec):
    """Same entity seen 'in different words': add an orthogonal-ish perturbation."""
    pert = NOISE_FRAC * F.normalize(torch.randn(D), dim=0)
    return F.normalize(vec + pert, dim=0)


def filler():
    return F.normalize(torch.randn(D), dim=0)


def build_stream(entities):
    """
    Interleaved stream of (vector, salience, label) events. label is the
    ground-truth entity id (0..4) or -1 for filler. Each entity appears
    MENTIONS_PER_ENTITY times; first mention is clean, the rest are noisy.
    ~50% of events are filler. Order is shuffled but reproducible.
    """
    events = []
    for eid, vec in enumerate(entities):
        for m in range(MENTIONS_PER_ENTITY):
            v = vec.clone() if m == 0 else noisy(vec)
            sal = 1.0 + 0.1 * torch.randn(1).item()       # entity salience ~1
            events.append({"v": v, "sal": sal, "label": eid, "first": (m == 0)})
    n_filler = len(events)                                  # ~50% filler
    for _ in range(n_filler):
        sal = 0.0 + 0.1 * torch.randn(1).item()            # filler salience ~0
        events.append({"v": filler(), "sal": sal, "label": -1, "first": False})

    perm = torch.randperm(len(events))
    stream = [events[i] for i in perm.tolist()]
    return stream


# --------------------------------------------------------------- recovery ---
def cosine_read(M_occ, query):
    slots = F.normalize(M_occ, dim=-1)
    q = F.normalize(query, dim=-1)
    attn = torch.softmax((slots @ q) / TEMP, dim=-1)
    return attn @ M_occ


def recovery_matrix(ctrl, entities):
    occ_idx = ctrl.occ.nonzero(as_tuple=True)[0]
    M_occ = ctrl.M[occ_idx]                                 # (n_occupied, d)
    R = torch.zeros(N_ENT, N_ENT)
    for i, qi in enumerate(entities):
        readback = cosine_read(M_occ, qi)
        for j, ej in enumerate(entities):
            R[i, j] = F.cosine_similarity(readback, ej, dim=0)
    return R


# ------------------------------------------------------------------ run -----
def run_config(name, stream, entities, use_gate):
    ctrl = HeuristicWriteController(D, SLOTS, repeat_thresh=REPEAT_THRESH,
                                    gate_thresh=0.5, update=0.5, use_gate=use_gate)
    ctrl.reset()

    entity_first_slot = {}     # entity id -> slot of its FIRST occurrence (stream order)
    merges_correct = merges_wrong = 0
    alloc_wrong = 0            # first occurrence that did NOT cleanly allocate
    filler_allocated = filler_skipped = 0
    overflow = 0
    repeat_cos_samples = []

    for ev in stream:
        res = ctrl.step(ev["v"], salience=ev["sal"])
        if res["action"] == "overflow":
            overflow += 1
            continue
        label = ev["label"]

        if label == -1:                                    # filler ground truth
            if res["action"] == "skip":
                filler_skipped += 1
            elif res["action"] == "new":
                filler_allocated += 1
            continue

        # entity ground truth — "first" defined by STREAM ORDER (the stream is
        # shuffled, so a noisy mention may legitimately precede the clean one).
        if label not in entity_first_slot:                 # FIRST time we see this entity
            if res["action"] == "new":
                entity_first_slot[label] = res["slot"]
            else:                                          # collapsed onto another entity
                entity_first_slot[label] = res["slot"]
                alloc_wrong += 1
        else:                                              # a REPEAT -> must route to its slot
            if res["max_cos"] is not None:
                repeat_cos_samples.append(res["max_cos"])
            want = entity_first_slot[label]
            if res["action"] == "repeat" and res["slot"] == want:
                merges_correct += 1
            else:
                merges_wrong += 1

    # slots that ground-truth entities ended up on
    entity_slots = sorted(set(entity_first_slot.values()))
    R = recovery_matrix(ctrl, entities)

    return {
        "ctrl": ctrl, "R": R, "use_gate": use_gate,
        "n_alloc": ctrl.n_alloc, "entity_slots": entity_slots,
        "entity_first_slot": entity_first_slot,
        "merges_correct": merges_correct, "merges_wrong": merges_wrong,
        "alloc_wrong": alloc_wrong,
        "filler_allocated": filler_allocated, "filler_skipped": filler_skipped,
        "overflow": overflow,
        "repeat_cos_mean": (sum(repeat_cos_samples) / len(repeat_cos_samples))
        if repeat_cos_samples else float("nan"),
    }


def report(name, r):
    total_repeats = r["merges_correct"] + r["merges_wrong"]
    print(f"\n{'='*70}\nCONFIG: {name}\n{'='*70}")
    print(f"  true distinct entities      : {N_ENT}")
    print(f"  slots allocated (total)     : {r['n_alloc']}"
          f"   (entities={len(r['entity_slots'])}, "
          f"filler={r['filler_allocated']}, overflow={r['overflow']})")
    print(f"  entity->slot map            : "
          f"{ {NAMES[k]: v for k, v in sorted(r['entity_first_slot'].items())} }")
    print(f"  first-mention mis-allocations: {r['alloc_wrong']}  (entities collapsed at first sight)")
    print(f"  noisy repeats merged        : {r['merges_correct']}/{total_repeats} correct"
          f"   (wrongly split/misrouted: {r['merges_wrong']})")
    print(f"  mean cos(noisy repeat, slot): {r['repeat_cos_mean']:.3f}  "
          f"(threshold {REPEAT_THRESH})")
    print(f"  filler                      : skipped={r['filler_skipped']}, "
          f"allocated={r['filler_allocated']}")

    R = r["R"]
    print(f"\n  RECOVERY MATRIX  R[i,j] = cos(read-back querying i, entity j)")
    head = "   query \\ recov ".ljust(18) + "".join(f"{n:>8}" for n in NAMES)
    print(head); print("  " + "-" * (len(head) - 2))
    for i in range(N_ENT):
        row = f"  {NAMES[i]:<16}"
        for j in range(N_ENT):
            row += f"{R[i, j].item():>8.3f}"
        ok = torch.argmax(R[i]).item() == i
        row += "  self✓" if ok else "  OTHER✗"
        print(row)

    diag = R.diagonal()
    off = R[~torch.eye(N_ENT, dtype=torch.bool)]
    all_self = all(torch.argmax(R[i]).item() == i for i in range(N_ENT))
    margin = (diag.min() - off.max()).item()
    one_slot_each = ((len(r["entity_slots"]) == N_ENT)
                     and (r["merges_wrong"] == 0) and (r["alloc_wrong"] == 0))
    distinct = all_self and margin > 0.10
    print(f"\n  diag min={diag.min():.3f} mean={diag.mean():.3f} | "
          f"off-diag max={off.max():.3f} | margin={margin:+.3f}")
    print(f"  => one slot per entity, no wrong splits? {'YES ✓' if one_slot_each else 'NO ✗'}")
    print(f"  => all 5 recovered DISTINCTLY?           {'YES ✓' if distinct else 'NO ✗'}")
    return one_slot_each, distinct


def main():
    torch.manual_seed(SEED)
    entities = make_entities()

    # sanity: entities ~orthogonal
    cs = [F.cosine_similarity(entities[a], entities[b], dim=0).item()
          for a in range(N_ENT) for b in range(a + 1, N_ENT)]
    print(f"Entity pairwise cosine: max|cos|={max(abs(c) for c in cs):.3f} (≈0 ⇒ distinct)")

    stream = build_stream(entities)
    n_ent_ev = sum(1 for e in stream if e["label"] != -1)
    print(f"Stream length={len(stream)}  entity-events={n_ent_ev}  "
          f"filler={len(stream) - n_ent_ev}  "
          f"({MENTIONS_PER_ENTITY} mentions × {N_ENT} entities, ~50% filler)")

    r_gate = run_config("GATED (salience write-gate ON — realistic)", stream, entities, True)
    r_pure = run_config("UNGATED (pure content heuristic, per spec)", stream, entities, False)
    g_slots, g_dist = report("GATED  — salience gate rejects filler", r_gate)
    p_slots, p_dist = report("UNGATED — pure content novelty", r_pure)

    # learned controller: just confirm it runs (forward + one backward)
    print(f"\n{'='*70}\nLEARNED controller smoke-test (runs only)\n{'='*70}")
    lc = LearnedWriteController(D, SLOTS)
    Mtest = torch.randn(SLOTS, D)
    gate, logits = lc(entities[0].unsqueeze(0), Mtest)
    loss = (1 - gate).mean() + F.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    print(f"  forward OK: gate shape {tuple(gate.shape)}, slot_logits {tuple(logits.shape)}; "
          f"backward OK (loss={loss.item():.3f})")

    print(f"\n{'#'*70}\nVERDICT\n{'#'*70}")
    print(f"  GATED   : {'1-slot-per-entity ✓' if g_slots else 'fragmented ✗'} | "
          f"{'5 recovered distinctly ✓' if g_dist else 'not separable ✗'} | "
          f"filler skipped {r_gate['filler_skipped']}/"
          f"{r_gate['filler_skipped']+r_gate['filler_allocated']}")
    print(f"  UNGATED : {'1-slot-per-entity ✓' if p_slots else 'fragmented ✗'} | "
          f"{'5 recovered distinctly ✓' if p_dist else 'not separable ✗'} | "
          f"filler allocated {r_pure['filler_allocated']} extra slots")
    print()
    if g_slots and g_dist:
        print("  => With a salience gate, the content-similarity controller SEGMENTS")
        print("     and ASSIGNS correctly from a messy interleaved stream: one slot")
        print("     per entity, noisy repeats merged, filler rejected, all 5 recoverable.")
    if p_dist and r_pure["filler_allocated"] > 0:
        print("  => Pure content novelty also merges repeats and keeps the 5 entities")
        print("     recoverable, but CANNOT tell a new entity from a new filler (both")
        print(f"     novel) — it burned {r_pure['filler_allocated']} extra slots on filler. The new-vs-filler")
        print("     call is the one piece that needs the gate / a learned entity-")
        print("     detector — that is the part worth training next.")
    elif not p_dist:
        print("  => Even with extra filler slots the entities blurred — investigate.")


if __name__ == "__main__":
    main()
