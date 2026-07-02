"""
test_separable_write.py — SRM write-mechanism diagnostic (NO training, NO language).

Core question
-------------
Can a NON-BLENDING ("separable") write keep distinct entities individually
recoverable from Spider Web's Solar Ring Memory (SRM), where the current
BLENDING write cannot?

This is a pure mechanism unit test — seconds, not hours. There is zero training
and zero language. We hand-construct 3 distinct random entity vectors
("ball", "Lily", "key"), write them into a fresh memory under each write_mode,
then probe recoverability:

  * query the memory with each entity's OWN vector,
  * read it back with a PLAIN cosine-similarity attention over slots
    (NO learned / randomly-initialised projections — so we isolate the WRITE
    policy itself, not any untrained read weights),
  * measure cosine(read-back, each original entity).

Interpretation
--------------
  separable  : querying "ball" reads back ~ball -> high self-cosine on the
               diagonal, low cross-cosine off-diagonal, argmax on the diagonal.
               All 3 entities recovered DISTINCTLY.
  blend      : every slot becomes the same superposition, so the read-back is
               query-independent mush -> diagonal ~ off-diagonal, argmax NOT
               reliably on the diagonal. Entities NOT separable.

If separable recovers all 3 and blend does not, the write design is sound and
training is worth it. If even separable cannot recover 3 hand-written vectors,
the mechanism is broken regardless of training.

Run:  python3 test_separable_write.py
"""

import torch
import torch.nn.functional as F

from config import Config, ModelConfig, MemoryConfig
from core.memory import SolarRingMemory

D      = 64          # entity / slot dimension
SLOTS  = 8           # memory slots (>= number of entities)
TEMP   = 0.05        # read softmax temperature (sharp content-addressing)
SEED   = 0
ENTITIES = ["ball", "Lily", "key"]


def build_memory(write_mode):
    """Fresh SRM in the requested write_mode, plus a fresh (B=1) memory bank."""
    cfg = Config(
        model=ModelConfig(dim=D),
        memory=MemoryConfig(slots=SLOTS, alpha=0.9, beta=0.1, write_mode=write_mode),
    )
    mem = SolarRingMemory(cfg).eval()
    m_t, _ = mem.init_memory(batch_size=1, device=torch.device("cpu"))
    # start from an EMPTY bank so we read only what we write (no seed noise);
    # init_memory still resets the round-robin write pointer for us.
    m_t = torch.zeros_like(m_t)
    return mem, m_t


def cosine_read(M, query):
    """
    Plain content-addressable read (no trained weights):
      score_i = cos(query, slot_i) / TEMP ;  attn = softmax ;  readback = sum attn_i * slot_i
    M     : (1, SLOTS, D)
    query : (D,)
    returns readback (D,)
    """
    slots = F.normalize(M[0], dim=-1)              # (SLOTS, D)
    q     = F.normalize(query, dim=-1)             # (D,)
    scores = (slots @ q) / TEMP                    # (SLOTS,)
    attn   = torch.softmax(scores, dim=-1)         # (SLOTS,)
    return attn @ M[0]                             # (D,)


@torch.no_grad()
def run_mode(write_mode, entities):
    mem, M = build_memory(write_mode)

    # --- WRITE the 3 entities, one per step ---
    for vec in entities:
        # payload shape (B=1, D); write() returns updated bank + advanced pointer
        M, mem._write_ptr = mem.write(M, vec.unsqueeze(0), mem._write_ptr)

    # --- PROBE: query with each entity, read back, cosine to every entity ---
    n = len(entities)
    R = torch.zeros(n, n)                           # R[i,j] = cos(readback(query_i), entity_j)
    for i, q in enumerate(entities):
        readback = cosine_read(M, q)
        for j, e in enumerate(entities):
            R[i, j] = F.cosine_similarity(readback, e, dim=0)
    return M, R


def report(write_mode, R):
    n = R.shape[0]
    print(f"\n{'='*64}\nWRITE MODE: {write_mode!r}\n{'='*64}")
    print("Recovery matrix  R[i,j] = cosine( read-back when querying i ,  entity j )")
    print("(rows = the query entity, cols = which entity the read-back matches)\n")
    header = "query \\ recovers".ljust(18) + "".join(f"{e:>9}" for e in ENTITIES)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{ENTITIES[i]:<18}"
        for j in range(n):
            v = R[i, j].item()
            cell = f"{v:>9.3f}"
            row += cell
        argmax_ok = torch.argmax(R[i]).item() == i
        row += "   <- top=self ✓" if argmax_ok else "   <- top=OTHER ✗"
        print(row)

    diag = R.diagonal()
    off  = R[~torch.eye(n, dtype=torch.bool)]
    all_self = all(torch.argmax(R[i]).item() == i for i in range(n))
    margin = (diag.min() - off.max()).item()
    print(f"\n  diag (self-recovery)  min={diag.min():.3f}  mean={diag.mean():.3f}")
    print(f"  off-diag (cross-talk) max={off.max():.3f}  mean={off.mean():.3f}")
    print(f"  separation margin (min_diag - max_offdiag) = {margin:+.3f}")
    distinct = all_self and margin > 0.10
    print(f"  => all 3 recovered DISTINCTLY?  {'YES ✓' if distinct else 'NO ✗'}")
    return distinct


def main():
    torch.manual_seed(SEED)
    # 3 distinct random entities (unit-norm, ~orthogonal in expectation)
    entities = [F.normalize(torch.randn(D), dim=0) for _ in ENTITIES]

    # sanity: how (dis)similar are the raw entities to each other?
    print("Raw entity pairwise cosine (should be ~0, they are distinct):")
    for a in range(len(entities)):
        for b in range(a + 1, len(entities)):
            c = F.cosine_similarity(entities[a], entities[b], dim=0).item()
            print(f"  cos({ENTITIES[a]}, {ENTITIES[b]}) = {c:+.3f}")

    _, R_blend = run_mode("blend", entities)
    _, R_sep   = run_mode("separable", entities)
    ok_blend = report("blend", R_blend)
    ok_sep   = report("separable", R_sep)

    print(f"\n{'#'*64}\nVERDICT\n{'#'*64}")
    print(f"  blend     : {'recovers 3 distinctly ✓' if ok_blend else 'cannot separate entities ✗'}")
    print(f"  separable : {'recovers 3 distinctly ✓' if ok_sep   else 'cannot separate entities ✗'}")
    if ok_sep and not ok_blend:
        print("\n  => Non-blending write KEEPS entities recoverable where blend does not.")
        print("     The write design is sound; training it is worth it.")
    elif ok_sep and ok_blend:
        print("\n  => Both modes separable at this tiny scale (no blend pressure yet).")
    elif not ok_sep:
        print("\n  => Even separable cannot recover 3 hand-written entities — the")
        print("     mechanism itself is broken, independent of training.")


if __name__ == "__main__":
    main()
