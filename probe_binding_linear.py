"""
DECISIVE DIAGNOSTIC — is the bound entity LINEARLY present in the residual stream
at the recall position?  (frozen model, train a 4-way linear probe)

We freeze the trained hybrid full-lookback checkpoint and, for many binding
examples of the form

    "<Name> had a <attr> <ENTITY>.  <fillers that never mention it>.  ... the"

feed only the PREFIX up to the recall cue "the" (the ENTITY word is NEVER in the
input, so there is no surface leakage). At that final position the model must
already "know" which of {ball, hat, key, car} was bound earlier. We extract the
residual-stream vector x there at four stages and train a linear probe (logistic
regression) to recover the entity, evaluating on HELD-OUT examples.

Stages (early -> late), all via forward hooks, no loop surgery:
  A embed+RoPE      : position/cue only  (CONTROL — entity is NOT at this token,
                      so this should sit near chance; validates no leakage)
  B post-rings      : what the chaos routing produced (input to the attention)
  C post-attention  : after the hybrid lookback had its chance to copy it back
  D post-final_norm : exactly what the lm_head reads

Chance = 25%.
  probe >> chance (>~60%) at C/D  -> entity IS linearly present; binding fails
                                     from loss pressure / the head -> FIX THE LOSS.
  probe ~ chance (~25%) at B/C/D  -> entity is NOT in the stream; the substrate
                                     destroyed it before retrieval -> ARCHITECTURE.

Also prints a one-line empirical confirmation of what the .detach() at
web.py:156 does to gradient flow (read -> write) in the memory path.

Run:  python3 probe_binding_linear.py
"""

import os
import sys
import random
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from config import get_config
from core.web import SpiderWeb

# default = the original best arm; pass a path to probe a different checkpoint
# (e.g. checkpoints/substrate_fix/best.pt for the post-fix re-measure).
CKPT     = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hybrid_full/best.pt"
# cheap isolation: "ablate_sep" forces write_mode=blend at inference (separable
# read OFF) while keeping whatever no_meanpool the checkpoint trained with, to
# see whether no_meanpool ALONE moves the probe.
ABLATE_SEP = len(sys.argv) > 2 and sys.argv[2] == "ablate_sep"
N_PER    = 175                                  # examples per entity
SEED     = 0
N_SPLITS = 3
TEST_FRAC = 0.30

ENTITIES = ["ball", "hat", "key", "car"]
FEMALE = ["Lily", "Sara", "Mia", "Anna", "Lucy", "Emma", "Zoe", "Nina", "Ruby", "Ella"]
MALE   = ["Tom", "Ben", "Max", "Sam", "Leo", "Jack", "Tim", "Finn", "Jake", "Noah"]
ATTRS  = ["red", "blue", "green", "yellow", "big", "small", "soft", "old",
          "new", "shiny", "tiny", "fluffy", "round", "warm", "bright"]
VERBS  = ["threw", "held", "dropped", "kept", "hugged", "grabbed", "washed", "painted"]
INTRO  = ["{name} had a {attr} {obj}.", "{name} found a {attr} {obj}.",
          "{name} got a {attr} {obj}.", "One day {name} saw a {attr} {obj}."]
FILLER = ["{pron} went to the park.", "{pron} played in the yard.",
          "{pron} ran down the hill.", "{pron} sat by the tree.",
          "{pron} sang a happy song.", "{pron} walked to school.",
          "It was a sunny day.", "The sun was warm and bright.",
          "{pron} laughed and smiled.", "{pron} jumped up and down.",
          "{pron} took a little nap.", "Birds flew in the sky."]
# recall prefixes that END at "the" (entity would come next; no attr in between)
RECALL_PREFIX = ["Then {name} {verb} the", "{name} wanted the",
                 "{Pron} picked up the", "At last {name} found the",
                 "{Pron} held the", "Then {name} {verb} the"]


def build_examples():
    rng = random.Random(SEED)
    exs = []
    for label, obj in enumerate(ENTITIES):
        for _ in range(N_PER):
            if rng.random() < 0.5:
                name, pron, Pron = rng.choice(FEMALE), "she", "She"
            else:
                name, pron, Pron = rng.choice(MALE), "he", "He"
            f = dict(name=name, pron=pron, Pron=Pron,
                     attr=rng.choice(ATTRS), obj=obj, verb=rng.choice(VERBS))
            parts = [rng.choice(INTRO).format(**f)]
            for _ in range(rng.randint(1, 3)):
                parts.append(rng.choice(FILLER).format(**f))
            parts.append(rng.choice(RECALL_PREFIX).format(**f))  # ends at "the"
            exs.append((" ".join(parts), label))
    rng.shuffle(exs)
    return exs


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    cfg = get_config()
    cfg.model.dim = 64; cfg.model.hidden_dim = 256; cfg.model.max_seq_len = 256
    cfg.model.use_hybrid = True; cfg.model.lookback_width = -1   # full lookback
    cfg.memory.slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    # match the trained substrate-fix architecture if this is a fixed checkpoint
    fixes = ckpt.get("fixes", "")
    cfg.model.sharp_head      = "sharp" in fixes
    cfg.model.residual_stream = "residual" in fixes
    cfg.model.undetach_mem    = "undetach" in fixes
    cfg.model.no_meanpool     = ("nomeanpool" in fixes) or bool(ckpt.get("no_meanpool", False))
    wm = ckpt.get("write_mode", "blend")
    if ABLATE_SEP:
        wm = "blend"           # isolation: separable read OFF at inference
    cfg.memory.write_mode = wm
    # ── GENERIC flag restore: rebuild the EXACT architecture/forward path the
    # checkpoint was trained with. Any checkpoint-meta key that names a
    # ModelConfig field is copied over, coerced to that field's type. This
    # replaces hand-listing flags — the pattern that silently DROPPED
    # name_lookback and ran the wrong architecture at eval (locRec 0% while
    # training showed 0.99). A NEW training flag using the same name now
    # auto-restores here, so the bug class can't recur on a future checkpoint.
    #   Skipped: `no_meanpool` (encoded in `fixes`, set above via OR) and
    #   `write_mode` (lives in cfg.memory, set above). None values keep the
    #   ModelConfig default (older checkpoints store unset flags as None).
    for k, v in ckpt.items():
        if k in ("model", "optimizer", "lr_sched", "no_meanpool") or v is None:
            continue
        if hasattr(cfg.model, k):
            cur = getattr(cfg.model, k)
            setattr(cfg.model, k,
                    type(cur)(v) if isinstance(cur, (bool, int, float)) else v)
    model = SpiderWeb(cfg).to(device)
    miss, unexp = model.load_state_dict(state, strict=False)
    new_pref = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj", "copy_gate", "name_lookback",
                "conc_gate")
    bad = [k for k in miss if not k.startswith(new_pref)]
    assert not bad and not unexp, f"load mismatch: missing={bad} unexpected={unexp}"
    print(f"[probe] arch flags: sharp_head={cfg.model.sharp_head} "
          f"residual_stream={cfg.model.residual_stream} "
          f"undetach_mem={cfg.model.undetach_mem} "
          f"no_meanpool={cfg.model.no_meanpool} "
          f"write_mode={cfg.memory.write_mode}"
          f"{' [ABLATE_SEP]' if ABLATE_SEP else ''}  fixes={fixes!r}", flush=True)
    model.eval()
    return model, device, ckpt.get("ema_ce")


@torch.no_grad()
def extract(model, device, sp, examples):
    """Run each example (B=1) and grab the last-position residual at 4 stages."""
    cap = {}
    h = []
    h.append(model.rope.register_forward_hook(
        lambda m, i, o: cap.__setitem__("A", o.detach())))
    h.append(model.hybrid_lookback.register_forward_pre_hook(
        lambda m, i: cap.__setitem__("B", i[0].detach())))
    h.append(model.hybrid_lookback.register_forward_hook(
        lambda m, i, o: cap.__setitem__("C", o[0].detach())))
    h.append(model.final_norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("D", o.detach())))

    feats = {s: [] for s in "ABCD"}
    labels = []
    for text, label in examples:
        ids = sp.EncodeAsIds(text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        cap.clear()
        model(inp, tau=0.1, hard=True)
        for s in "ABCD":
            feats[s].append(cap[s][0, -1].float().cpu())   # last position
        labels.append(label)
    for hk in h:
        hk.remove()
    feats = {s: torch.stack(v) for s, v in feats.items()}
    return feats, torch.tensor(labels)


def train_probe(Xtr, ytr, Xte, yte, epochs=400, lr=0.05, wd=1e-2):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = nn.Linear(Xtr.size(1), len(ENTITIES))
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        tr = (clf(Xtr).argmax(1) == ytr).float().mean().item()
        te = (clf(Xte).argmax(1) == yte).float().mean().item()
    return tr, te


def detach_confirmation():
    print("\n" + "=" * 72)
    print("DETACH CONFIRMATION — web.py:156  m_t_flat_full[mask] = out['m_t'].detach()")
    print("=" * 72)
    # WITHOUT detach: gradient from a later 'read' reaches the 'write' producer.
    w = torch.tensor([2.0], requires_grad=True)
    written = w * 3.0                       # a "write" produced from params (w)
    read_loss = (written * 5.0).sum()       # a later "read" uses it
    read_loss.backward()
    g_keep = w.grad.item()
    # WITH detach (as in the model): the write is severed from the graph.
    w2 = torch.tensor([2.0], requires_grad=True)
    written2 = (w2 * 3.0).detach()          # exactly what .detach() does at :156
    read_loss2 = (written2 * 5.0).sum()
    read_loss2.backward() if written2.requires_grad else None
    g_det = None if w2.grad is None else w2.grad.item()
    print(f"  no-detach: read-loss backward gives write-producer grad = {g_keep}")
    print(f"  detached : read-loss backward gives write-producer grad = {g_det}  "
          f"(written.requires_grad={written2.requires_grad})")
    print("  => .detach() makes the stored memory a NON-differentiable scratchpad:")
    print("     gradient from a future READ cannot flow back to shape the WRITE.")
    print("     The model can never learn WHAT to store from the read payoff — the")
    print("     exact failure mode of every memory-based binding attempt.")


def main():
    torch.manual_seed(SEED); random.seed(SEED)
    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    model, device, ema = load_model()
    exs = build_examples()
    print(f"[probe] ckpt={CKPT} ema_ce={ema} device={device} "
          f"examples={len(exs)} ({N_PER}/entity) entities={ENTITIES}", flush=True)
    print(f"[probe] e.g. -> {exs[0][0]!r}  (label={ENTITIES[exs[0][1]]})", flush=True)

    feats, labels = extract(model, device, sp, exs)
    N = len(labels)
    n_te = int(N * TEST_FRAC)

    stage_name = {"A": "embed+RoPE (control)", "B": "post-rings (pre-attn)",
                  "C": "post-attention", "D": "post-final_norm (head input)"}
    print(f"\n{'stage':<32} {'train':>8} {'test (held-out)':>16}   chance=25%")
    print("-" * 64)
    results = {}
    for s in "ABCD":
        tr_accs, te_accs = [], []
        for k in range(N_SPLITS):
            g = torch.Generator().manual_seed(100 + k)
            perm = torch.randperm(N, generator=g)
            te_idx, tr_idx = perm[:n_te], perm[n_te:]
            tr, te = train_probe(feats[s][tr_idx], labels[tr_idx],
                                 feats[s][te_idx], labels[te_idx])
            tr_accs.append(tr); te_accs.append(te)
        tr_m = sum(tr_accs) / N_SPLITS
        te_m = sum(te_accs) / N_SPLITS
        te_sd = (sum((a - te_m) ** 2 for a in te_accs) / N_SPLITS) ** 0.5
        results[s] = te_m
        print(f"{stage_name[s]:<32} {100*tr_m:>7.1f}% "
              f"{100*te_m:>9.1f}% ± {100*te_sd:>3.1f}", flush=True)

    print("-" * 64)
    late = max(results["C"], results["D"])
    print(f"\nDECISIVE READ (best of post-attention / head-input = {100*late:.1f}%):")
    if late > 0.60:
        print("  >> Entity identity IS linearly present at the recall site.")
        print("  >> Binding fails from LACK OF LOSS PRESSURE / the output head,")
        print("     NOT missing information.  => FIX THE LOSS.")
    elif late < 0.40:
        print("  >> Entity identity is NOT recoverable at the recall site (~chance).")
        print("  >> The substrate (chaos routing + RMSNorm + spectral norm + the")
        print("     detached writes) destroyed it before retrieval. => ARCHITECTURE.")
    else:
        print("  >> PARTIAL: entity is weakly present. Both loss pressure and")
        print("     substrate degradation are contributing. Report numbers.")
    print(f"  trajectory A->D: " +
          " ".join(f"{s}={100*results[s]:.0f}%" for s in "ABCD"))

    detach_confirmation()


if __name__ == "__main__":
    main()
