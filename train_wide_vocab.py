"""
Step 2 — train attention-supervised retrieval over a WIDE (197-noun) training
vocab, so the QK cannot memorize a small key set and is forced toward a general
"copy the recently-introduced noun" rule.

Reuses the EXACT machinery / config that produced the narrow result (the only
changed variable is the entity vocabulary breadth, so the held-out attn-mass is
directly comparable to the narrow run's 0.23). no_meanpool stays False to match
that working run.

  python3 train_wide_vocab.py --steps 4000
"""
import json, argparse
import torch
import train_attn_super as TA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", type=str, default="checkpoints/wide_vocab")
    ap.add_argument("--w_attn", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_pointer", action="store_true",
                    help="enable copy/gen pointer readout (NLL on mixture)")
    ap.add_argument("--base", type=str, default=TA.BASE_CKPT,
                    help="warm-start checkpoint (e.g. checkpoints/wide_vocab/best.pt)")
    ap.add_argument("--warm_gen", action="store_true",
                    help="A: warm generative branch from trained lm_head")
    ap.add_argument("--lambda_floor", type=float, default=0.0,
                    help="B: floor on generative gate weight lambda")
    ap.add_argument("--w_router", type=float, default=0.0,
                    help="C: weight on the copy-gate router BCE")
    ap.add_argument("--p_bind", type=float, default=None,
                    help="D: fraction of binding examples (default 0.75)")
    ap.add_argument("--mem_copy", action="store_true",
                    help="source the copy from the orbital memory read (separable "
                         "store + embed-similarity emit) instead of a hybrid attn map")
    ap.add_argument("--multi_entity", action="store_true",
                    help="multi-entity selective binding (cue-conditioned target)")
    ap.add_argument("--oracle_bind", action="store_true",
                    help="ORACLE: hard-wire slot key/query to clause subject token id "
                         "(bypass content router) — localizes addressing vs value/emit")
    a = ap.parse_args()

    vocab = json.load(open("data/wide_vocab.json"))
    TA.OBJECTS = vocab["train"]                       # <-- wide training vocab
    print(f"[wide] training OBJECTS = {len(TA.OBJECTS)} distinct nouns "
          f"(narrow run used ~20). p_bind={TA.P_BIND} use_pointer={a.use_pointer}",
          flush=True)

    for bs in (48, 32, 24, 16):
        try:
            TA.run(bs, a.w_attn, a.out, a.steps, resume=a.resume,
                   use_pointer=a.use_pointer, base_ckpt=a.base,
                   warm_gen=a.warm_gen, lambda_floor=a.lambda_floor,
                   w_router=a.w_router, p_bind=a.p_bind, mem_copy=a.mem_copy,
                   multi_entity=a.multi_entity, oracle_bind=a.oracle_bind); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[wide] OOM at batch={bs}, falling back.", flush=True)
    print("[wide] OOM even at batch=16.")

if __name__ == "__main__":
    main()
