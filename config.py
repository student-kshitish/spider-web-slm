from dataclasses import dataclass, field
import torch


# -------------------------
# MODEL CONFIG
# -------------------------
@dataclass
class ModelConfig:
    dim: int = 96
    hidden_dim: int = 384
    num_rings: int = 4
    nodes_per_ring: int = 8
    vocab_size: int = 5000
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    use_query_read: bool = False   # orbital content-addressable memory read (off by default)
    use_struct_read: bool = False  # position-resolved structural retrieval (off by default)
    struct_mask: bool = True       # direction-gate the structural read (only if use_struct_read)
    # ── Unified hybrid: Spider Web routing + surgical gated lookback attention ──
    use_hybrid: bool = False       # master flag. OFF == pure Spider Web baseline
    lookback_width: int = 32       # bounded window (last N); <=0 -> full lookback
    # ── Pointer / copy readout (predicted fix for the emit-generalization gap) ──
    # OFF (default) == current behaviour: a per-token linear lm_head + CE. ON adds
    # a copy branch that emits the token actually present at the attended source
    # position (zero per-token emit params), ties the generative branch to the
    # embedding table, and mixes the two via a learned gate (NLL on the mixture).
    # Requires use_hybrid (reads hybrid_stats["attn"]). See core/web.py forward.
    use_pointer: bool = False
    # ── Pointer integration fixes (A/B) — stop the gate collapsing onto copy ────
    # A: warm-start the generative branch from the TRAINED lm_head instead of the
    #    untrained embed.weight tie, so generation is competent at step 0 (CE ~2.7,
    #    not ~17) and the gate has no reason to abandon it.
    pointer_warm_gen: bool = False
    # B: floor the generative weight lambda (affine remap lam -> floor+(1-floor)*lam)
    #    so the generative branch always receives gradient and can't be fully
    #    starved even if the router supervision is imperfect. 0.0 == no floor.
    pointer_lambda_floor: float = 0.0
    # ── mem_copy: source the copy from Spider Web's OWN orbital memory ──────────
    # OFF (default): the copy distribution is the hybrid lookback ATTENTION map
    # scattered by source-token id (an attention head). ON: the copy is sourced
    # from the SEPARABLE entity memory's content-keyed slot-routing read
    # (sep_stats["read_dist"]) transporting the source token EMBEDDINGS, then
    # decoded by a tied-embedding similarity readout. Requires write_mode=
    # "separable". This makes the orbital memory — not an attention map — the copy
    # engine. mem_copy_scale sharpens the cosine-similarity emit softmax.
    mem_copy: bool = False
    mem_copy_scale: float = 12.0
    # ── conc_gate: re-source the copy gate from READ-CONCENTRATION statistics ────
    # The generalization probe showed the default gate copy_gate(x) degrades with
    # lexical familiarity (lambda 0.49->0.83 from one novel word) because x carries
    # token-shaped features. Read-concentration (how peaked the copy distribution
    # copy_attn is) is lexicon-invariant and is the true correlate of copy success
    # (P_copy top-5 = 100% whenever read mass > 0.5). This flag conditions lambda on
    # [read_max, read_entropy] instead of / in addition to x. Values:
    #   "off"     : default gate, lambda = sigmoid(copy_gate(x))            (unchanged)
    #   "stats"   : lambda = sigmoid(conc_gate([read_max, read_entropy]))   (2-dim in)
    #   "stats_x" : lambda = sigmoid(conc_gate([read_max, read_entropy, x])) (2+d in)
    # Only meaningful with use_pointer; stats are taken from the copy distribution
    # (mem_copy: read_dist; else the hybrid attn map). The lambda floor still applies.
    conc_gate: str = "off"
    # ── Substrate fixes (diagnosed by probe_binding_linear.py) ──────────────────
    # Each isolates one cause of the bound entity being destroyed before the
    # recall position. All default OFF so existing checkpoints are unaffected.
    undetach_mem: bool = False     # (1) let gradient flow read->write in SRM (web.py:156)
    residual_stream: bool = False  # (2) additive hop highway: x = x + node_out (carries token identity)
    sharp_head: bool = False       # (3) drop spectral_norm on FFN/aux/lm_head (sharp copy circuits & logits)
    oracle_bind: bool = False      # ORACLE localization probe: hard-wire SeparableMemoryRead's
                                   #     slot key/query to the clause SUBJECT token id (one-hot on
                                   #     subj_id % slots), bypassing the learned content router.
                                   #     Tests whether selective-binding failure is addressing
                                   #     (fixable) vs downstream value/emit (deeper). Value unchanged.
    id_key_addr: bool = False      # IDENTITY-KEY ADDRESSING: source SeparableMemoryRead's
                                   #     write_key/read_query from the raw token EMBEDDING
                                   #     (embed(input_ids), pre-RoPE, pre-mixing) instead of the
                                   #     post-mixing hidden state x, where subject identity is
                                   #     smeared. Gives write and read a stable discrete anchor to
                                   #     agree on so the LEARNED router can form a subject-keyed
                                   #     address on its own. Value path still uses x (content).
    name_transport: bool = False   # NAME-TRANSPORT ORACLE (key-source probe; needs id_key_addr):
                                   #     at each object-intro position use the embedding of that
                                   #     clause's SUBJECT-NAME token as the key, and at the recall
                                   #     position use the CUE-NAME token's embedding (positions
                                   #     supplied via name_src). Tests whether learned addressing
                                   #     failed only because the key was read from the wrong token
                                   #     (embed("ball")/embed("the") carry no subject identity).
                                   #     Other positions keep the default id_key source.
    name_lookback: bool = False    # NAME-LOOKBACK (Step 2 — removes the last oracle):
                                   #     a LEARNED causal single-head attention (core.hybrid.
                                   #     NameLookback) that LOCATES the governing name itself
                                   #     instead of being handed its position (name_src). At
                                   #     each position t it forms name_hat[t] = sum_u a[t,u] *
                                   #     embed(input_ids[u]) — an IDENTITY-PRESERVING transport
                                   #     of raw embeddings (mirrors the mem_copy path), NOT a
                                   #     mix of hidden states — and feeds name_hat as the id_key
                                   #     to SeparableMemoryRead's write_key/read_query. The
                                   #     lookback attention a is exposed on-graph (name_stats
                                   #     ["attn"]) and supervised toward the known name positions
                                   #     (L_name, train only). name_src is used ONLY in that aux
                                   #     loss — the forward pass NEVER reads it, so at inference
                                   #     the model must find the name on its own. Supersedes the
                                   #     name_transport oracle (only changed variable: how the
                                   #     name reaches the key — gather -> learned lookback).
    # ── Step 6: write-policy primitives (native slot-memory ops, NTM-style) ─────
    # The Step-5 finding was that the separable store is ADDITIVE accumulation with
    # no ordering and no deletion. These add exactly those two write policies to
    # SeparableMemoryRead's (closed-form) slot memory; both default to a no-op so
    # warm-started checkpoints are unaffected.
    write_decay: float = 1.0       # DECAY-ON-WRITE (recency): when position u writes to
                                   #     slot s with strength a=write_w*gate, existing
                                   #     content is scaled by (1-(1-γ)a) BEFORE the add, so
                                   #     newer writes dominate within a slot. γ=1.0 -> no
                                   #     decay (identity, the Step-5 behavior). Folds into the
                                   #     read as a per-slot cumulative decay (exact, keeps the
                                   #     per-position read_dist diagnostic).
    erase: bool = False            # TARGETED ERASE (NTM-style): a SECOND learned name-lookback
                                   #     (web.py self.giver_lookback) resolves the GIVER; a
                                   #     zero-init erase gate e=sigmoid(erase_gate(x)) applies
                                   #     slot <- slot*(1 - e*w_giver) at the (learned) transfer
                                   #     position, suppressing the giver's slot for later reads.
                                   #     Zero-init -> erase starts OFF (warm-compatible).
    no_meanpool: bool = False      # (4) DON'T collapse the T axis when writing node memory
                                   #     (web.py:172). Registry stays position-resolved
                                   #     (B,T,slots,d) so an entity written at position k is
                                   #     not averaged across the whole sequence (irreversible).
                                   #     WARNING: (B,T,slots,d) on the autograd graph across all
                                   #     hops/nodes is ~T x memory — use a small batch.


# -------------------------
# MEMORY (SRM v2.1)
# -------------------------
@dataclass
class MemoryConfig:
    slots: int = 32
    alpha: float = 0.9
    beta: float = 0.1
    learned_init: bool = True
    # Write policy for the SRM:
    #   "blend"     : M_new = alpha*M_old + beta*payload  (current; all slots mix)
    #   "separable" : write each entity to its OWN slot (round-robin), occupied
    #                 slots are protected (no decay, no blend).
    write_mode: str = "blend"


# -------------------------
# LORENZ / CHAOS
# -------------------------
@dataclass
class LorenzConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 2.667
    dt: float = 0.005
    clamp: float = 15.0
    eps_init: float = 0.01


# -------------------------
# ROUTING
# -------------------------
@dataclass
class RoutingConfig:
    temp_start: float = 2.0
    temp_end: float = 0.1
    anneal_steps: int = 8000
    max_hops: int = 6


# -------------------------
# TRAINING (FINAL FIXED)
# -------------------------
@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2.5e-4
    lr_min: float = 1e-5
    weight_decay: float = 0.001
    grad_clip: float = 1.0
    warmup_steps: int = 200
    steps: int = 30000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 REQUIRED BY train_main.py
    use_bf16: bool = True
    use_compile: bool = False   # set False first (compile can break early runs)

    # Optional dtype (safe default)
    dtype: torch.dtype = torch.bfloat16

    # -------------------------
    # LOSS WEIGHTS
    # -------------------------
    entropy_weight: float = 0.003
    balance_weight: float = 0.001


# -------------------------
# MAIN CONFIG
# -------------------------
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    lorenz: LorenzConfig = field(default_factory=LorenzConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    return Config()