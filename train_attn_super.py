"""
Step (retrieval supervision) — teach the lookback QK circuit to retrieve.

Diagnosis (probe_source_vs_recall.py + probe_copy_path.py): the bound entity is
100% linearly present at its OWN position through the head, and the lookback copy
hardware is live (gate ~0.8, o_proj ~1.9, contributes ~40% of the residual) — but
the recall token "the" attends to the SOURCE entity with only ~2-4% mass (rank
~10). The QK match never learned the retrieval. So supervise it directly.

AUXILIARY (flag-gated by --w_attn > 0)
--------------------------------------
For each synthetic binding example we know:
  - source_pos : where the entity was introduced ("... a red BALL ...")
  - recall_pos : the "the" token whose next-token target IS that entity
The lookback attention at recall_pos is a distribution over earlier positions u.
We add a cross-entropy that maximises attn[recall_pos -> source_pos]:

  attn_loss = - mean_over_binding_examples  log attn[ b, recall_pos_b, source_pos_b ]
  total     = CE(+entropy/balance)  +  w_attn * attn_loss

w_attn = 0 -> pure control (identical run, no retrieval pressure). The attn tensor
is exposed on-graph by HybridLookbackAttention (stats["attn"]).

TRAINING (recall-heavy, real budget)
------------------------------------
  - warm from checkpoints/substrate_fix/best.pt (undetach+residual+sharp kept ON,
    blend memory, mean-pool ON — the source probe shows the substrate preserves
    the entity, so no architecture change).
  - p_bind = 0.75 (binding-heavy so retrieval pressure isn't diluted by
    unconstrained TinyStories "the ___").
  - FULL lookback (lookback_width = -1) so recall can reach any earlier source.
  - >= 4000 steps (growing a new retrieval head).

Usage:
  python3 train_attn_super.py --w_attn 0.5 --out checkpoints/attn_super
  python3 train_attn_super.py --w_attn 0.0 --out checkpoints/attn_super_off
"""
import os, sys, argparse, random
os.environ["WANDB_MODE"] = "disabled"
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import (Config, ModelConfig, MemoryConfig, LorenzConfig,
                    RoutingConfig, TrainConfig)
from core.web import SpiderWeb
from train.loss import SpiderWebLoss
from train.scheduler import get_cosine_warmup_scheduler, TemperatureScheduler

T_LEN      = 64
P_BIND     = 0.75
BASE_CKPT  = "checkpoints/substrate_fix/best.pt"
TS_PATH    = "data/raw/tinystories.txt"
PAD_ID     = 0
IGNORE     = -1                      # SpiderWebLoss CE ignore_index
NEW_PREFIXES = ("hybrid_lookback", "separable_mem", "query_read",
                "struct_read", "recall_proj", "copy_gate")

# ── synthetic binding vocabulary (superset; probe tests ball/hat/key/car) ──
OBJECTS = ["ball","hat","cat","dog","book","cup","kite","doll","box","drum",
           "frog","fish","cake","key","shoe","car","bell","duck","bear","boat"]
FEMALE = ["Lily","Sara","Mia","Anna","Lucy","Emma","Zoe","Nina","Ruby","Ella"]
MALE   = ["Tom","Ben","Max","Sam","Leo","Jack","Tim","Finn","Jake","Noah"]
ATTRS  = ["red","blue","green","yellow","big","small","soft","old","new","shiny",
          "tiny","fluffy","round","warm","bright"]
VERBS  = ["threw","held","dropped","kept","hugged","grabbed","washed","painted"]
INTRO  = ["{name} had a {attr} {obj}.","{name} found a {attr} {obj}.",
          "{name} got a {attr} {obj}.","One day {name} saw a {attr} {obj}."]
FILLER = ["{pron} went to the park.","{pron} played in the yard.",
          "{pron} ran down the hill.","{pron} sat by the tree.",
          "{pron} sang a happy song.","{pron} walked to school.",
          "It was a sunny day.","The sun was warm and bright.",
          "{pron} laughed and smiled.","{pron} jumped up and down.",
          "{pron} took a little nap.","Birds flew in the sky."]
# recall ends with the OBJECT as the final token (its next-token target)
RECALL = ["Then {name} {verb} the {obj}","{name} wanted the {obj}",
          "{Pron} picked up the {obj}","At last {name} found the {obj}",
          "{Pron} held the {obj}","Then {name} {verb} the {obj}"]

# ── Multi-entity binding (decisive selective-binding test) ──────────────────────
# k distinct subjects, each bound to a distinct object; a recall cue NAMES one
# subject (must use {name}, never a pronoun — a pronoun would be ambiguous across
# subjects). The supervision target is the CUE subject's object intro, so neither
# "copy the recent entity" nor "copy the first entity" wins. Toggled by --multi_entity.
MULTI_ENTITY        = False
N_ENTITIES_CHOICES  = [2, 3]
MULTI_RECALL = ["Then {name} {verb} the {obj}", "{name} wanted the {obj}",
                "At last {name} {verb} the {obj}", "Later {name} {verb} the {obj}"]


def ft_config(batch_size, slots, steps, use_pointer=False,
              warm_gen=False, lambda_floor=0.0, mem_copy=False,
              mem_copy_scale=12.0, write_mode="blend", no_meanpool=False,
              oracle_bind=False, id_key=False, name_transport=False) -> Config:
    return Config(
        model=ModelConfig(dim=64, hidden_dim=256, num_rings=4, nodes_per_ring=8,
                          vocab_size=5000, max_seq_len=256,
                          use_hybrid=True, lookback_width=-1,          # FULL lookback
                          sharp_head=True, residual_stream=True,
                          undetach_mem=True, no_meanpool=no_meanpool,
                          use_pointer=use_pointer,                     # copy/gen mixture arm
                          pointer_warm_gen=warm_gen,                   # A: gen<-lm_head
                          pointer_lambda_floor=lambda_floor,           # B: lambda floor
                          mem_copy=mem_copy,                           # copy<-orbital memory
                          mem_copy_scale=mem_copy_scale,
                          oracle_bind=oracle_bind,                     # subject-keyed oracle
                          id_key_addr=id_key,                          # identity-key addressing
                          name_transport=name_transport),              # name-transport key oracle
        memory=MemoryConfig(slots=slots, alpha=0.9, beta=0.1, write_mode=write_mode),
        lorenz=LorenzConfig(),
        routing=RoutingConfig(temp_start=0.3, temp_end=0.1,
                              anneal_steps=steps, max_hops=6),
        train=TrainConfig(batch_size=batch_size, lr=1e-4, lr_min=1e-5,
                          weight_decay=1e-3, grad_clip=1.0, warmup_steps=50,
                          steps=steps, use_bf16=True, use_compile=False,
                          entropy_weight=0.05, balance_weight=0.001),
    )


class StructuredMix(Dataset):
    """Binding examples (with known source/recall positions) mixed with
    TinyStories blocks. Generative -> idx ignored; order set by self.rng so an
    ON vs OFF pair with the same seed sees IDENTICAL data."""

    def __init__(self, sp, seed, n=200_000):
        self.sp = sp
        self.rng = random.Random(seed)
        self.n = n
        with open(TS_PATH, encoding="utf-8") as f:
            self.ts = sp.EncodeAsIds(f.read())
        self.n_ts = len(self.ts) // (T_LEN + 1)

    def __len__(self):
        return self.n

    def _pad(self, inp, tgt, recall_pos, source_pos, is_bind, subj=None, name_src=None):
        L = len(inp)
        x = torch.full((T_LEN,), PAD_ID, dtype=torch.long)
        y = torch.full((T_LEN,), IGNORE, dtype=torch.long)
        s = torch.full((T_LEN,), -1, dtype=torch.long)         # oracle subject ids (-1 = none)
        ns = torch.full((T_LEN,), -1, dtype=torch.long)        # name-transport source positions (-1 = default key)
        x[:L] = torch.tensor(inp, dtype=torch.long)
        y[:L] = torch.tensor(tgt, dtype=torch.long)
        if subj is not None:                                   # {position: subject_token_id}
            for p, sid in subj.items():
                if 0 <= p < T_LEN:
                    s[p] = sid
        if name_src is not None:                               # {position: name_token_position}
            for p, npos in name_src.items():
                if 0 <= p < T_LEN and 0 <= npos < T_LEN:
                    ns[p] = npos
        return x, y, recall_pos, source_pos, is_bind, s, ns

    def _binding(self):
        r = self.rng
        for _ in range(8):                       # resample if too long
            if r.random() < 0.5:
                name, pron, Pron = r.choice(FEMALE), "she", "She"
            else:
                name, pron, Pron = r.choice(MALE), "he", "He"
            obj = r.choice(OBJECTS)
            f = dict(name=name, pron=pron, Pron=Pron,
                     attr=r.choice(ATTRS), obj=obj, verb=r.choice(VERBS))
            parts = [r.choice(INTRO).format(**f)]
            for _ in range(r.randint(1, 3)):
                parts.append(r.choice(FILLER).format(**f))
            parts.append(r.choice(RECALL).format(**f))
            text = " ".join(parts)
            pcs = self.sp.encode(text, out_type=str)
            ids = self.sp.EncodeAsIds(text)
            occ = [i for i, p in enumerate(pcs) if p.replace("▁", "").lower() == obj]
            if len(occ) < 2:
                continue                          # entity must appear twice
            last = occ[-1]
            if last != len(ids) - 1:
                continue                          # obj must be the final token
            inp, tgt = ids[:-1], ids[1:]          # standard LM shift
            recall_pos = last - 1                 # the "the" token
            source_pos = occ[0]                   # intro occurrence
            if len(inp) <= T_LEN and source_pos < recall_pos:
                return self._pad(inp, tgt, recall_pos, source_pos, 1)
        # fallback: minimal example
        return self._binding_minimal()

    def _binding_minimal(self):
        text = "Lily had a red ball. Then Lily threw the ball"
        pcs = self.sp.encode(text, out_type=str); ids = self.sp.EncodeAsIds(text)
        occ = [i for i, p in enumerate(pcs) if p.replace("▁", "").lower() == "ball"]
        return self._pad(ids[:-1], ids[1:], occ[-1] - 1, occ[0], 1)

    def _binding_multi(self):
        """k=2..3 entities, each bound to a distinct subject; a recall cue NAMES
        one subject. The supervision target source_pos is the CUE subject's object
        intro (cue-conditioned), NOT the most-recent entity — so a recency/position
        shortcut cannot satisfy it. cue is uniform over the k subjects."""
        r = self.rng
        for _ in range(12):                          # resample on bad tokenization
            k = r.choice(N_ENTITIES_CHOICES)
            if len(OBJECTS) < k:
                k = 2
            names = r.sample(FEMALE + MALE, k)       # distinct subjects
            objs  = r.sample(OBJECTS, k)             # distinct objects
            prons = [("she", "She") if nm in FEMALE else ("he", "He")
                     for nm in names]
            parts = []
            for i in range(k):                       # introduce each binding
                f = dict(name=names[i], pron=prons[i][0], Pron=prons[i][1],
                         attr=r.choice(ATTRS), obj=objs[i], verb=r.choice(VERBS))
                parts.append(r.choice(INTRO).format(**f))
            for _ in range(r.randint(1, 2)):         # generic filler
                parts.append(r.choice(FILLER).format(pron=prons[0][0],
                                                     Pron=prons[0][1]))
            cue = r.randrange(k)                     # UNIFORM cue (not always last)
            cf = dict(name=names[cue], pron=prons[cue][0], Pron=prons[cue][1],
                      obj=objs[cue], verb=r.choice(VERBS))
            parts.append(r.choice(MULTI_RECALL).format(**cf))   # NAME-based recall
            text = " ".join(parts)
            pcs = self.sp.encode(text, out_type=str)
            ids = self.sp.EncodeAsIds(text)
            cobj = objs[cue]
            occ = [i for i, p in enumerate(pcs)
                   if p.replace("▁", "").lower() == cobj]
            if len(occ) != 2:                        # cue obj: intro + final only
                continue
            last = occ[-1]
            if last != len(ids) - 1:                 # cue obj must be final token
                continue
            inp, tgt = ids[:-1], ids[1:]
            recall_pos = last - 1                    # the "the" token
            source_pos = occ[0]                      # CUE subject's intro -> target
            if len(inp) <= T_LEN and source_pos < recall_pos:
                # ORACLE labels: subject token id at each entity's object intro
                # and at the recall position (cued subject). Same subject -> same
                # id -> same slot in oracle_bind. (no-op unless --oracle_bind.)
                #
                # NAME-TRANSPORT labels: the POSITION of the governing name token to
                # source the slot key from — the clause's SUBJECT name for each
                # object-intro, and the CUE name for the recall. Found by the name's
                # first sub-piece (space-prefixed, e.g. "▁Lily"/"▁Fin"), taking the
                # occurrence immediately BEFORE the keyed position (its own clause /
                # the recall clause). (no-op unless --name_transport.)
                def _name_pos_before(nm, before):
                    fp = self.sp.encode(nm, out_type=str)[0]      # first piece of the name
                    cand = [j for j, p in enumerate(pcs) if p == fp and j < before]
                    return cand[-1] if cand else None
                subj, name_src = {}, {}
                for i in range(k):
                    oi = [j for j, p in enumerate(pcs)
                          if p.replace("▁", "").lower() == objs[i]]
                    if oi and oi[0] < len(inp):
                        subj[oi[0]] = self.sp.EncodeAsIds(names[i])[0]
                        npos = _name_pos_before(names[i], oi[0])    # this clause's subject name
                        if npos is not None:
                            name_src[oi[0]] = npos
                subj[recall_pos] = self.sp.EncodeAsIds(names[cue])[0]
                cpos = _name_pos_before(names[cue], recall_pos)     # the cue name at recall
                if cpos is not None:
                    name_src[recall_pos] = cpos
                return self._pad(inp, tgt, recall_pos, source_pos, 1, subj, name_src)
        return self._binding_minimal()

    def _ts(self):
        b = self.rng.randrange(self.n_ts)
        s = b * (T_LEN + 1)
        chunk = self.ts[s: s + T_LEN + 1]
        return self._pad(chunk[:-1], chunk[1:], -1, -1, 0)

    def __getitem__(self, idx):
        if self.rng.random() < P_BIND:
            return self._binding_multi() if MULTI_ENTITY else self._binding()
        return self._ts()


def attn_supervision(out, recall_pos, source_pos, is_bind, attn=None):
    """CE pulling the copy-source distribution attn[recall_pos]->source_pos for
    binding examples. `attn` defaults to the hybrid lookback map; for mem_copy the
    caller passes the orbital-memory read sep_stats['read_dist'] instead, so the
    SAME retrieval supervision trains the memory read (the RETRIEVE link).
    Returns (loss, realized_mass, n_binding)."""
    attn = (out["hybrid_stats"]["attn"] if attn is None else attn).float()  # (B,T,T)
    bi = is_bind.bool().nonzero(as_tuple=True)[0]
    if bi.numel() == 0:
        z = attn.sum() * 0.0
        return z, 0.0, 0
    rp, sp_ = recall_pos[bi], source_pos[bi]
    rows = attn[bi, rp, :]                               # (n, T)
    logp = torch.log(rows.clamp_min(1e-9))
    loss = F.nll_loss(logp, sp_)
    with torch.no_grad():
        mass = rows[torch.arange(bi.numel(), device=rows.device), sp_].mean().item()
    return loss, mass, bi.numel()


def addr_supervision(out, recall_pos, subj, S):
    """L_addr — pull the LEARNED slot routing toward the subject-keyed slot, the
    exact target the oracle hard-wires, but as SUPERVISION (oracle OFF at
    inference). Converts the oracle from a runtime override into a training
    teacher for the content router:

        L_addr = CE(write_w[obj_intro], subj_id % S)     # writes land on subject slot
               + CE(read_w [recall_pos], subj_id % S)    # read returns to SAME slot

    subj (B,T): subject token id at each entity's object-intro position AND at the
    recall position (cued subject), -1 elsewhere (built by StructuredMix._binding_multi).
    write term = all labeled intros (subj>=0 and NOT the recall pos); read term =
    the recall pos. Uses out["sep_stats"]["write_w"/"read_w"] (on-graph).
    Returns (loss, write_slot_acc, read_slot_acc, n_write, n_read)."""
    ss = out.get("sep_stats")
    if ss is None or "write_w" not in ss:
        z = torch.zeros((), device=subj.device)
        return z, 0.0, 0.0, 0, 0
    ww = ss["write_w"].float()                       # (B,T,S)
    rw = ss["read_w"].float()                        # (B,T,S)
    B, T, _ = ww.shape
    ar   = torch.arange(B, device=subj.device)
    tgt  = (subj.clamp_min(0) % S)                   # (B,T) subject-keyed slot
    read_mask = torch.zeros(B, T, dtype=torch.bool, device=subj.device)
    read_mask[ar, recall_pos] = True                 # recall_pos may be -1 for TS rows...
    read_mask &= (subj >= 0)                          # ...masked out: TS rows have subj all -1
    write_mask = (subj >= 0) & ~read_mask             # entity object-intros only

    loss = ww.sum() * 0.0
    w_acc = r_acc = 0.0
    n_w = int(write_mask.sum().item()); n_r = int(read_mask.sum().item())
    if n_w > 0:
        lw = torch.log(ww[write_mask].clamp_min(1e-9))
        loss = loss + F.nll_loss(lw, tgt[write_mask])
        with torch.no_grad():
            w_acc = (ww[write_mask].argmax(-1) == tgt[write_mask]).float().mean().item()
    if n_r > 0:
        lr = torch.log(rw[read_mask].clamp_min(1e-9))
        loss = loss + F.nll_loss(lr, tgt[read_mask])
        with torch.no_grad():
            r_acc = (rw[read_mask].argmax(-1) == tgt[read_mask]).float().mean().item()
    return loss, w_acc, r_acc, n_w, n_r


def gate_router_loss(out, recall_pos, is_bind, targets):
    """FIX C — supervise the copy gate lambda as a ROUTER so copy fires ONLY on the
    'emit the bound entity' token and generation handles all other language.

      target lambda = 0 (COPY) at the recall position of each binding example
      target lambda = 1 (GEN)  at every other valid (non-pad) position

    Returns (loss, lam_at_recall_mean). The two BCE terms are averaged separately
    so the dense generate positions don't drown the ~1-per-example copy signal."""
    pt = out.get("pointer", None)
    if pt is None:
        return torch.zeros((), device=targets.device), float("nan")
    lam = pt["lambda"].float()                                  # (B,T) gen weight
    valid = (targets != IGNORE)                                 # real positions
    copy_mask = torch.zeros_like(lam, dtype=torch.bool)
    bi = is_bind.bool().nonzero(as_tuple=True)[0]
    if bi.numel() > 0:
        copy_mask[bi, recall_pos[bi]] = True
    copy_mask &= valid
    gen_mask = valid & ~copy_mask
    lam_c = lam.clamp(1e-6, 1 - 1e-6)
    gen_term  = (-torch.log(lam_c)[gen_mask].mean()
                 if gen_mask.any() else lam.sum() * 0.0)        # push -> 1 (gen)
    copy_term = (-torch.log(1 - lam_c)[copy_mask].mean()
                 if copy_mask.any() else lam.sum() * 0.0)       # push -> 0 (copy)
    lam_recall = (lam[copy_mask].mean().item() if copy_mask.any() else float("nan"))
    return gen_term + copy_term, lam_recall


def run(batch_size, w_attn, out_dir, steps, resume=False,
        use_pointer=False, base_ckpt=BASE_CKPT,
        warm_gen=False, lambda_floor=0.0, w_router=0.0, p_bind=None,
        mem_copy=False, multi_entity=False, oracle_bind=False,
        w_addr=0.0, id_key=False, oracle_anneal=False, name_transport=False):
    torch.manual_seed(42); random.seed(123)
    global P_BIND, MULTI_ENTITY
    if p_bind is not None:
        P_BIND = p_bind                              # FIX D: rebalance bind vs LM mass
    MULTI_ENTITY = multi_entity                      # multi-entity selective binding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mem_copy bundles the WRITE link: separable (non-blending) entity store +
    # position-resolved memory (no time mean-pool), so the copy can be sourced
    # from the memory's own content-keyed read instead of a hybrid attn map.
    write_mode = "separable" if mem_copy else "blend"
    no_meanpool = bool(mem_copy)
    # anneal implies the oracle is on (as the distillation teacher); alpha 1->0.
    if oracle_anneal:
        oracle_bind = True
    # name-transport keys ARE identity-key addressing (embed source) with a
    # name-substitution at the write/read positions, so it implies id_key.
    if name_transport:
        id_key = True

    ckpt_path = f"{out_dir}/last.pt" if resume else base_ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in ckpt["model"].items()}
    slots = state["rings.0.0.memory.m_t_seed"].shape[0]
    cfg = ft_config(batch_size, slots, steps, use_pointer=use_pointer,
                    warm_gen=warm_gen, lambda_floor=lambda_floor,
                    mem_copy=mem_copy, write_mode=write_mode,
                    no_meanpool=no_meanpool, oracle_bind=oracle_bind,
                    id_key=id_key, name_transport=name_transport)
    start_step = int(ckpt.get("step", 0)) if resume else 0

    model = SpiderWeb(cfg).to(device)
    m, u = model.load_state_dict(state, strict=False)
    bad = [k for k in m if not k.startswith(NEW_PREFIXES)]
    assert not bad and not u, f"warm load: missing={bad} unexpected={u}"
    assert next(model.parameters()).dtype == torch.float32
    print(f"[as] device={device} batch={batch_size} slots={slots} steps={steps} "
          f"w_attn={w_attn} use_pointer={use_pointer} out={out_dir}  "
          f"{'RESUME '+ckpt_path if resume else 'WARM from '+base_ckpt}", flush=True)
    print(f"[as] INTEGRATION: warm_gen={warm_gen} lambda_floor={lambda_floor} "
          f"w_router={w_router} p_bind={P_BIND}", flush=True)
    print(f"[as] MEM_COPY={mem_copy} (copy source = "
          f"{'orbital memory read_dist + embed-similarity emit' if mem_copy else 'hybrid attn scatter'}"
          f"; write_mode={write_mode} no_meanpool={no_meanpool})", flush=True)
    print(f"[as] MULTI_ENTITY={MULTI_ENTITY} "
          f"(k={N_ENTITIES_CHOICES if MULTI_ENTITY else 1}; supervision target = "
          f"{'CUE subject intro (selective)' if MULTI_ENTITY else 'single entity'})",
          flush=True)
    print(f"[as] ORACLE_BIND={oracle_bind} "
          f"(slot key/query hard-wired to SUBJECT token id, bypass content router)"
          if oracle_bind else "[as] ORACLE_BIND=False (learned content router)",
          flush=True)
    print(f"[as] L_ADDR: w_addr={w_addr} id_key={id_key} oracle_anneal={oracle_anneal} "
          f"(supervise LEARNED write_w/read_w -> subj%{slots}; "
          f"key from {'TOKEN EMBEDDING (identity)' if id_key else 'hidden state x (smeared)'}; "
          f"{'oracle alpha 1->0 over training' if oracle_anneal else 'oracle OFF at inference'})",
          flush=True)
    print(f"[as] NAME_TRANSPORT={name_transport} "
          f"(key at write/read positions = SUBJECT-/CUE-NAME token embedding, "
          f"gathered via name_src; corrects the key-source bug)"
          if name_transport else "[as] NAME_TRANSPORT=False (key = token's own embedding)",
          flush=True)

    loss_fn = SpiderWebLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay,
                                  betas=(0.9, 0.95), fused=(device.type == "cuda"))
    lr_sched = get_cosine_warmup_scheduler(optimizer, cfg)
    tau_sched = TemperatureScheduler(cfg)
    if resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"]); lr_sched.load_state_dict(ckpt["lr_sched"])
    on_cuda = device.type == "cuda"
    ac = (torch.autocast("cuda", torch.bfloat16)
          if on_cuda and cfg.train.use_bf16 else nullcontext())

    sp = spm.SentencePieceProcessor(); sp.Load("data/tokenizer.model")
    ds = StructuredMix(sp, seed=12345)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=0, drop_last=True)        # num_workers=0 -> identical A/B order
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:    return next(data_iter)
        except StopIteration:
            data_iter = iter(loader); return next(data_iter)

    os.makedirs(out_dir, exist_ok=True)
    model.train()
    ema = float(ckpt["ema_ce"]) if resume and "ema_ce" in ckpt else None
    best = ema if ema is not None else float("inf")
    step0 = None

    def save(path, step, ema_val):
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "lr_sched": lr_sched.state_dict(),
                    "step": step, "ema_ce": ema_val,
                    "fixes": "undetach+residual+sharp"
                             + ("+nomeanpool" if no_meanpool else ""), "w_attn": w_attn,
                    "use_pointer": use_pointer, "pointer_warm_gen": warm_gen,
                    "pointer_lambda_floor": lambda_floor, "w_router": w_router,
                    "mem_copy": mem_copy, "write_mode": write_mode,
                    "multi_entity": MULTI_ENTITY,
                    "oracle_bind": oracle_bind,
                    "id_key_addr": id_key, "w_addr": w_addr,
                    "oracle_anneal": oracle_anneal,
                    "name_transport": name_transport}, path)

    print(f"[as] {'Step':>6} {'CE':>8} {'EMA':>8} {'genCE':>7} {'attnL':>7} "
          f"{'mass':>6} {'lam':>6} {'lamRcl':>6} {'rout':>6} {'tau':>5} | "
          f"{'gnorm':>7}", flush=True)
    for step in range(start_step, steps):
        x, y, rp, sp_pos, isb, subj, nsrc = next_batch()
        x, y = x.to(device), y.to(device)
        rp, sp_pos, isb = rp.to(device), sp_pos.to(device), isb.to(device)
        subj = subj.to(device); nsrc = nsrc.to(device)
        tau = tau_sched.get_temp(step)
        # oracle anneal: alpha 1->0 linearly, handing routing to the learned router
        # while value/emit stay trained (distillation). alpha=1 unless annealing.
        alpha = (max(0.0, 1.0 - step / max(1, steps - 1)) if oracle_anneal else 1.0)
        with ac:
            out = model(x, tau=tau, hard=False, subj_id=subj, oracle_alpha=alpha,
                        name_src=nsrc)
            if on_cuda and cfg.train.use_bf16:
                out["logits"] = out["logits"].float()
            loss, mets = loss_fn(out, y, entropy_weight=cfg.train.entropy_weight,
                                 w_depth=0.0, w_recall=0.0)
            # RETRIEVE link: for mem_copy supervise the ORBITAL MEMORY read
            # (sep_stats["read_dist"]); otherwise the hybrid lookback map. Same
            # recall->source CE, applied to whichever distribution feeds the copy.
            copy_attn = (out["sep_stats"]["read_dist"] if mem_copy else None)
            attn_loss, mass, n_b = attn_supervision(out, rp, sp_pos, isb,
                                                    attn=copy_attn)
            if w_attn > 0:
                loss = loss + w_attn * attn_loss
            router_loss, lam_recall = gate_router_loss(out, rp, isb, y)
            if w_router > 0:
                loss = loss + w_router * router_loss
            # ADDRESS link: supervise the LEARNED slot routing toward subj%S (the
            # oracle target, as a teacher). Gated by w_addr; OFF at inference.
            addr_loss, addr_wacc, addr_racc, n_aw, n_ar = addr_supervision(out, rp, subj, slots)
            if w_addr > 0:
                loss = loss + w_addr * addr_loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[as] *** NaN/Inf at step {step}. Stopping. ***", flush=True)
            return True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step(); lr_sched.step()

        ce = mets["ce"]
        ema = ce if ema is None else 0.95 * ema + 0.05 * ce
        pt = out.get("pointer")
        lam_m = pt["lambda_mean"] if pt is not None else float("nan")
        # generative-only CE: NLL on P_gen alone -> proves FIX A warm-started the
        # generative branch (~2.7), independent of how the gate mixes in copy.
        if pt is not None:
            with torch.no_grad():
                lpg = torch.log(pt["P_gen"].float().clamp_min(1e-9))
                gen_ce = F.nll_loss(lpg.view(-1, lpg.size(-1)), y.view(-1),
                                    ignore_index=IGNORE).item()
        else:
            gen_ce = float("nan")

        rout_v = router_loss.item()

        def _line(s):
            print(f"[as] {s:>6} {ce:>8.4f} {ema:>8.4f} {gen_ce:>7.3f} "
                  f"{attn_loss.item():>7.3f} {mass:>6.3f} {lam_m:>6.3f} "
                  f"{lam_recall:>6.3f} {rout_v:>6.3f} {tau:>5.2f} | "
                  f"{float(gnorm):>7.2f}", flush=True)

        if step == start_step:
            step0 = ce; _line(step)
            print(f"[as]   step-{start_step} mixCE={ce:.3f}  genCE={gen_ce:.3f}  "
                  f"attn_mass={mass:.3f}  lambda_mean={lam_m:.3f}  "
                  f"lambda@recall={lam_recall:.3f}  router={rout_v:.3f}  "
                  f"(n_bind/batch={n_b})", flush=True)
            print(f"[as]   step-{start_step} L_addr={addr_loss.item():.3f} "
                  f"writeSlotAcc={addr_wacc:.3f} readSlotAcc={addr_racc:.3f} "
                  f"alpha={alpha:.3f} (n_write={n_aw} n_read={n_ar})", flush=True)
        elif step <= start_step + 50 and step % 10 == 0:
            _line(step)
        elif step % 250 == 0 or step == steps - 1:
            _line(step)
            if w_addr > 0:
                print(f"[as]   L_addr={addr_loss.item():.3f} writeSlotAcc={addr_wacc:.3f} "
                      f"readSlotAcc={addr_racc:.3f} alpha={alpha:.3f}", flush=True)
            save(f"{out_dir}/last.pt", step + 1, ema)
        if step > cfg.train.warmup_steps and ema < best:
            best = ema; save(f"{out_dir}/best.pt", step + 1, ema)

    save(f"{out_dir}/last.pt", steps, ema)
    print(f"[as] DONE w_attn={w_attn} step0_CE={step0:.3f} final_EMA={ema:.3f} "
          f"best_EMA={best:.3f} final_mass={mass:.3f} -> {out_dir}", flush=True)
    if w_addr > 0:
        print(f"[as] DONE L_addr: w_addr={w_addr} id_key={id_key} "
              f"final writeSlotAcc={addr_wacc:.3f} readSlotAcc={addr_racc:.3f} "
              f"(learned addressing; oracle OFF at inference)", flush=True)
    nanp = [n for n, p in model.named_parameters() if p.isnan().any()]
    print(f"[as] {'no NaN in weights' if not nanp else 'NaN weights: '+str(nanp)}",
          flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_attn", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="checkpoints/attn_super")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_pointer", action="store_true",
                    help="enable copy/gen pointer readout (NLL on mixture)")
    ap.add_argument("--base", type=str, default=BASE_CKPT,
                    help="warm-start checkpoint (default: substrate_fix)")
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
    ap.add_argument("--w_addr", type=float, default=0.0,
                    help="weight on L_addr: supervise learned write_w/read_w toward "
                         "subj%%S (the oracle target as a teacher). OFF at inference.")
    ap.add_argument("--id_key", action="store_true",
                    help="source slot key/query from the raw token EMBEDDING "
                         "(identity-preserving) instead of the smeared hidden state x")
    ap.add_argument("--oracle_bind", action="store_true",
                    help="hard-wire slot routing to subj token id (localization probe)")
    ap.add_argument("--oracle_anneal", action="store_true",
                    help="start oracle_bind ON and anneal alpha 1->0 (distill the "
                         "oracle teacher into the learned router); implies --oracle_bind")
    ap.add_argument("--name_transport", action="store_true",
                    help="key-source oracle: at write/read positions source the slot "
                         "key from the SUBJECT-/CUE-NAME token embedding (implies --id_key)")
    a = ap.parse_args()
    for bs in (48, 32, 24, 16):
        try:
            run(bs, a.w_attn, a.out, a.steps, resume=a.resume,
                use_pointer=a.use_pointer, base_ckpt=a.base,
                warm_gen=a.warm_gen, lambda_floor=a.lambda_floor,
                w_router=a.w_router, p_bind=a.p_bind, mem_copy=a.mem_copy,
                multi_entity=a.multi_entity, oracle_bind=a.oracle_bind,
                w_addr=a.w_addr, id_key=a.id_key,
                oracle_anneal=a.oracle_anneal,
                name_transport=a.name_transport); return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[as] OOM at batch={bs}, falling back.", flush=True)
    print("[as] OOM even at batch=16.")


if __name__ == "__main__":
    main()
