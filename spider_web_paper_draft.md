# Spider Web: A Non-Attention Small Language Model, and What Its Failure to Bind Reveals About Attention

**KSHITISH-BEHERA**

*Draft — work in progress. Numbers are measured from the experiments described; references are placeholders to be filled.*

---

## Abstract

We present **Spider Web**, a 3.48M-parameter small language model built without attention. In place of the attention matrix, Spider Web routes tokens through a topology of "rings" using a trainable Lorenz-63 chaotic dynamical system for the routing decision, stores context in a constant-size **Solar Ring Memory (SRM)**, and encodes position with a **3-Axis Polar RoPE**. Trained on TinyStories with a 5,000-token SentencePiece vocabulary, the model reaches an eval cross-entropy of **4.77** (50-batch average; 44% below the random baseline of log 5000 = 8.52) and produces locally fluent, grammatically well-formed text.

The central contribution of this paper is not the model's loss but a **systematic investigation of why it cannot bind** — the inability to remember an entity introduced earlier in a passage and re-emit it at a later recall site (coreference, attribute recall). We frame binding as a four-link causal chain — **WRITE → SURVIVE → RETRIEVE → DECODE** — and run a sequence of controlled A/B arms, each isolating one link with a linear-probe diagnostic. Every prior memory/retrieval mechanism repairs at most one link and the chain stays broken (0/4). Direct **retrieval supervision** repairs the chain end-to-end for the *trained* vocabulary (in-template top-5 recall **60%**, recall-site linear probe **90%**), but a wide-vocabulary generalization study with a **disjoint** held-out entity vocabulary then decomposes binding cleanly into two halves with opposite fates: the **copy/routing** half *generalizes* (held-out attention-to-source mass 0.23 → 0.83) while the **emit/readout** half *structurally cannot* (held-out emit rank 222/246, two-alternative forced choice 15.8% — below chance). We argue this decomposition is the load-bearing finding: binding requires a *content-agnostic, identity-preserving copy* operation at **both** the routing and the output stages — exactly what attention with a tied/pointer readout supplies natively, and exactly what an architecture built from mixing, averaging, and normalization operators structurally lacks. We demonstrate this not by argument but by building the alternative and measuring where it breaks.

---

## 1. Architecture

Spider Web is a deliberately non-Transformer language model. The design constraints (no attention, no LSTM/GRU, RMSNorm throughout, spectral-normalized projections, Gumbel-softmax routing) were fixed in advance as a project constitution; the architecture is the consequence of holding them.

**Topology.** Tokens are processed by a web of **4 rings × 8 nodes = 32 WebNodes**. A token enters at the outermost ring and, over at most **6 routing hops**, may *stay* in its ring, *jump inward* one ring, or *exit*. The intent is a radial hierarchy: outer rings handle transient/local computation, inner rings hold persistent state.

**Lorenz Router.** The 3-way routing decision (stay / jump-in / exit) is produced by `proj_in → 5-step Lorenz-63 ODE integration → proj_out → Gumbel-softmax`. The Lorenz system supplies a trainable nonlinear dynamical map for the decision rather than a plain linear classifier. A key empirical result of the project is that this chaotic map trains stably — no divergence, grad norm ~0.4 throughout the substrate-fix runs.

**Solar Ring Memory (SRM).** Each node owns a small bank of learnable memory slots (32 slots/node). The write is a gated convex blend `M_new = α·M_old + β·(σ(W_g(h)) ⊙ h)` followed by RMSNorm. Because the bank is fixed-size, the per-step activation cost is nominally **O(1) in sequence length**, in contrast to the O(T) KV cache / O(T²) score matrix of attention. (§4 tests whether this theoretical advantage manifests.)

**3-Axis Polar RoPE.** Position is encoded on three axes — temporal (sequence index), angular (node index within ring), and radial (ring index) — computed on the fly, so there is no fixed-length positional table.

**WebNode.** Each node is SwiGLU FFN (64 → 256 → 64) + lateral "ring attention" + SRM read/write + Lorenz router. The lateral attention is **not** content-based cross-token attention; it is a **ring-mean** of peer states (`neighbours = ring_states.mean(0)`). This distinction — mean-pooling rather than content-addressed selection — becomes central in §5.

**Readout.** A final RMSNorm feeds a single linear LM head to vocab logits. The head is **untied** from the input embedding (a separate `nn.Linear(d, vocab)`); §5.5 shows this is one of the two structural reasons emit cannot generalize.

**Scale and data.** dim=64, hidden=256, vocab=5,000 (SentencePiece), seq_len=128, ~3.48M parameters, trained on TinyStories (~43 MB) for 20,000 steps on a single 8 GB RTX 5050 laptop GPU.

---

## 2. Language-modeling result

| Metric | Value |
|---|---|
| Parameters | 3.48 M |
| Training steps | 20,000 |
| Random baseline CE | 8.52 (= log 5000) |
| Best single-batch CE | 4.42 (reported for completeness only) |
| **Eval CE — 50-batch average (headline)** | **4.77** |
| Reduction from random | 44% |

The honest headline is the 50-batch average of **4.77**; the 4.42 figure is a single lucky batch and is not representative of held-out performance. (A prior project session established the same lesson for an earlier run: a "3.73" target turned out to be the single luckiest of 2,709 batches; the true 50-batch average of those weights was 3.9996 — a cautionary note carried into how we report here.)

Qualitatively the model fills the correct grammatical slot (POS) for copula completions, places ~65% probability mass on frequent character nouns (`girl` 46.8%, `boy` 18.3% after "there was a little"), completes genre/register collocations, and learns surface patterns such as dialogue punctuation (64.6% on `,`). Within a single sentence it is locally competent; cross-sentence capabilities are the subject of §5.

A direct CE comparison to the published TinyStories baselines is **not valid**: that work uses a 10K GPT-2-derived vocabulary (CE is tokenizer-relative; random baseline log 10000 = 9.21 ≠ our 8.52) and reports GPT-4 quality scores rather than a numeric loss. We therefore do not claim a competitive CE; we report ours against its own random baseline only.

---

## 3. Implementation findings

Three bugs in this project were not incidental — each was a measured finding about how this class of architecture interacts with standard training machinery, and each materially changed behavior.

### 3.1 The bfloat16 master-weight freeze (silent, total)

Early training called `model.to(torch.bfloat16)`, storing **all** parameters as bf16 master weights. RMSNorm weights initialize to 1.0; AdamW updates are ~2×10⁻⁴; the unit of least precision for bf16 at magnitude 1.0 is 2⁻⁷ ≈ 7.8×10⁻³ — roughly **40× larger than the update**. Every step rounded to zero. All **97 RMSNorm instances** (3 per node × 32 nodes + final norm) were frozen for the entire run; loss stalled at CE 7–8 regardless of duration or learning rate.

*Caught by* sampling `final_norm.weight` at step 300 and observing exactly 0% movement across runs. *Fixed* by keeping master weights in fp32 and wrapping only forward+loss in `torch.autocast("cuda", torch.bfloat16)`. After the fix, sampled norm weights moved 2–4% off 1.0 by step 300 in every run; over 20,000 steps the final-norm weight drifted **274%** from initialization. This is a general hazard for RMSNorm-heavy, normalization-everywhere architectures trained in low precision.

### 3.2 The dead innermost ring

The exit condition included a `| (current_ring == 0)` term, so any token that reached the innermost ring was **immediately forced to exit** before it could be used. Ring 0 — intended as the persistent-state terminal of the radial hierarchy — was unreachable: **0 visits across all runs**. The very ring meant to hold bound entities never ran. Removing the term (exit now requires an explicit action-2 decision; inward movement remains guarded by `current_ring > 0`) made ring 0 reachable. With auxiliary depth-pressure and inner-ring recall losses added, **inner-ring usage rose 20% → 58%** and the recall reconstruction loss fell **1.00 → 0.80**, at a CE cost (~7.25 at 2,000 steps from the aux terms).

### 3.3 Detached memory writes (a non-differentiable scratchpad)

The SRM write detached its output (`out['m_t'].detach()`), so no gradient flowed from a downstream *read* back to the *write* that produced the stored content. Confirmed empirically: a future read's backward gives the write-producer `grad = None` with detach vs `grad ≈ 15.0` without. The consequence is fundamental: the model **cannot learn what to store** from the payoff of later reading it — the memory is a non-differentiable scratchpad, not a trained store. This is the single most important of the three for binding (§5), because link 1 (WRITE) is unlearnable by construction.

---

## 4. Scaling and saturation: the model is data-bound, not architecture-bound

Two independent observations indicate that, at this scale, the routing mechanism is **not** the binding/quality bottleneck — data volume and training budget are.

**Parameters are flat.** A dim=96 / 7.28M-param variant did not beat the 3.48M model at comparable budget (and earlier stalled at CE 7 before the bf16 fix). A planned ablation swapping the Lorenz router for an identically-structured **linear** router (same params, same budget) was interrupted, but the preliminary result at step 3,600 had the **linear router slightly ahead** in CE — i.e., the chaotic routing is unlikely to be where the performance lives at this scale.

**Memory scaling does not yet pay off.** SRM's headline property is O(1) activation cost in sequence length. Measured against a **parameter-matched** causal Transformer (dim=128, 4 heads, 11 layers, 3.45M params, 0.87% delta), over a 32× sequence increase (64 → 2048 tokens) Spider Web activation memory grew **7.8×** and the matched Transformer grew **7.3×** — nearly identical. The theoretical O(1) advantage **does not manifest** within the tested range, because the Transformer's T² scores are computed in bf16 and freed under `no_grad`, while Spider Web's per-node SRM expansions and routing bookkeeping accumulate comparable overhead. Throughput is far worse: the matched Transformer is **65–149× faster** across seq lengths, because the hop loop processes tokens sequentially ring-by-ring while the Transformer fuses the whole sequence in one kernel. This is an execution-model difference, not a tuning gap.

**Reading.** Both scaling axes are flat against a same-capacity Transformer, and a competent-baseline binding test (§5) rules out undertraining as the cause of the binding failure. The binding wall is therefore attributable to **architecture**, while the *quality* gap is attributable to **budget/data** — two separate conclusions the experiments keep distinct.

---

## 5. The binding investigation (centerpiece)

**Binding** is the ability to introduce an entity ("Lily had a red **ball**"), let intervening text pass without mentioning it, and then re-emit the specific earlier entity at a recall site ("She threw the ___" → **ball**). It is the prerequisite for coreference, attribute recall, and state tracking. Spider Web fails it; the value of this project is in localizing *why*, precisely.

### 5.1 Binding as a four-link chain

We decompose binding into a causal chain in which a break anywhere yields total failure:

1. **WRITE** — store the entity's identity when it is introduced.
2. **SURVIVE** — carry that identity, intact, across the intervening positions to the recall site.
3. **RETRIEVE / COPY** — at the recall site, select and copy the specific stored entity (not an average of context).
4. **DECODE / EMIT** — turn the retrieved identity into the correct output token.

Each prior arm of the investigation repaired *at most one* link; binding needs all four, so the score stayed **0/4** until the chain was supervised end-to-end.

### 5.2 Per-arm localization (each arm isolates one link)

A controlled-A/B method runs throughout: both arms warm-start from the competent CE-4.42 baseline (step-0 CE ≈ 5.2 confirms the language-competent model loaded, not a cold start), share seed and data order, and differ by exactly one mechanism. The decisive metric is **not** CE (which is near-neutral across arms) but a **binding probe** (is the specific earlier entity in top-5?) and a **frozen linear probe** (is the entity linearly recoverable at the recall site, 4-way over ball/hat/key/car, chance 25%).

- **Baseline substrate (link 2 broken).** Ring-mean lateral mixing + per-hop RMSNorm scrub token identity; no copy mechanism exists. The entity does not even survive to the recall site.

- **Fixed-lag inner-ring recall (link 1/3, negative).** An auxiliary loss reconstructs the embedding of the token seen `lag=4` positions back into inner-ring memory. The loss trains fine internally (**1.0 → 0.76**), but the A/B is null: Arm A (CE best 4.169) ≈ Arm B control (4.106); both emit a *generic* noun ("car") at the recall site, **neither retrieves the specific entity**. Fixed-lag reconstruction is not content-addressable, variable-distance retrieval.

- **Orbital query-read (link 3, negative).** A learned content-addressable read (query → scaled dot-product over all slots → per-ring bias → softmax → weighted slot sum → gated fuse). CE A 4.022 ≈ B 4.025 — no benefit, no binding. Tellingly, the learned per-ring bias **favored outer/transient rings** (realized mass [0.152, 0.220, 0.267, 0.361]) — the *opposite* of the persistence hypothesis. Diagnosed cause: the store is **mean-pooled over time** before reading (`m_t_new = …reshape(B,T,slots,d).mean(1)`), so the read queries a time-averaged bag and cannot isolate "the ball at position k." Content addressing cannot bind a store whose time axis has already collapsed.

- **Position-resolved structural read (link 3 hardware).** Keeping memory position-resolved (full T, no time-collapse) makes the read causal (zero mass above the diagonal) with lookback 0.96–0.99 — the *hardware* for retrieval now exists; CE stays neutral.

- **Hybrid gated lookback attention (link 3, learnable but not capable).** Bolting on a surgical gated content-attention path (Spider Web routing untouched) shows the gate **is** learnable: on-rate **15.8% → 91.4%**, o_proj norm 0.000 → 1.108, CE 5.24 → 4.43 — unlike the storage gate, which collapsed to ~0 (the lookback gate decides at *read* time with a *local* payoff; the storage gate decided at *write* time with a *distant* payoff and got no gradient). But learnability ≠ capability: binding probe **0/4**, and a frozen linear probe over depth is **at chance at every layer** (embed 25.9%, post-rings 24.0%, post-attention 24.3%, post-final-norm 23.7%; chance 25%). There is nothing clean to copy.

- **Substrate fixes — necessary, not sufficient.** Applying the three §3 repairs (un-detach writes, additive residual highway `x = x + node_out`, drop spectral-norm on the head/FFN) is stable (no NaN, grad norm ~0.4) and a large CE win (**3.90 → 2.93**; last-token pmax 0.06 → 0.40). The probe trajectory becomes **monotonic** A<B<C<D (26 < 27 < 29 < 31%) where before it was flat ~24% — entity identity is *faintly* more recoverable (head-input 30.6%, ~2.8σ) but still far below the >40–50% "survives" bar. CE gains again did not become binding.

### 5.3 The retrieval-supervision fix (chain repaired, in-distribution)

The probes show retrieval (link 3) as the live break: the query-key never learns to point at the source. We supervise it directly — an auxiliary NLL pulling the lookback attention at the recall position toward the source position (`w_attn = 0.5`), warm from the substrate-fix checkpoint. The effect is decisive and fast:

- attention-to-source mass: **ON 0.971** (argmax 98.7%) vs **OFF 0.067** (the original ~4% pathology reproduced);
- recall-site linear probe: **ON 90.0%** vs **OFF 26.7%** (chance 25%), with the jump localized exactly at the post-attention stage;
- CE 2.96 → 2.42; saturated by ~step 250.

The entity now reaches the recall site and is linearly present. A full-budget generation test (best checkpoint @ step 3033, CE EMA 2.96 → 2.547; gold entity *not* in the visible suffix, so inference uses trained weights only; n = 120/batch, chance top-5 ≈ 0.1%) then asks whether it is actually *decoded*:

| Test | top-1 | top-5 | median rank |
|---|---|---|---|
| **[A] in-template** (trained templates + vocab) | 15% | **60%** | 4 |
| **[C] novel sentence structure** (rel-clause/apposition, trained vocab) | 5% | 22% | 15 |
| **[B] unseen entity TYPES** (lamp/spoon/rope… never bound in training) | **0%** | **0%** | 462 |

In-template binding **decodes for the first time** (60% top-5 vs 0/4 for every prior arm) — a real result. But [B] is a complete failure to generalize: the supervised query-key learned **token-specific keys**, not a general "copy the recently introduced noun" rule.

### 5.4 The wide-vocabulary decomposition (the load-bearing finding)

To force the question, we train binding over a **wide** entity vocabulary — **197 training nouns vs 49 held-out, zero overlap** — with vocabulary breadth as the only variable (`w_attn = 0.5`). Measured on the disjoint held-out set, the two halves of binding split decisively:

**ROUTING / COPY generalizes** (the copy rule is relational and token-agnostic):
- held-out attention-to-source mass **0.23 (narrow) → 0.827 (wide)**; source = argmax **24% → 86.7%** — nearly matching trained. The query-key learned a general "copy the introduced noun" rule that fires on unseen nouns.

**EMIT / READOUT does not — and structurally cannot.** Two stacked failures:
1. **The copied payload is token-specific.** At the recall head-input, held-out identity is barely recoverable even by an *ideal tied* head (embedding-similarity rank **132/246 ≈ chance 123**). The value/output projections and the source ring-representation were fit on the 197 training nouns: they transport trained identities and garble unseen ones. Routing generalized; the value pathway **memorized**.
2. **The untied LM head has no output coordinate for unseen tokens.** A token's output direction in `lm_head` is shaped by gradient **only when that token appears as a target**. Held-out nouns therefore rank near the **bottom** (emit rank **222/246**, two-alternative forced choice **15.8% — below 50%**), worse even than embedding-similarity (222 vs 132).

Breadth also *dilutes* per-noun emit pressure ~50×, so even the **trained** decode collapsed (2AFC 65%, top-5 ~2% vs the narrow run's 56%) — more vocabulary cannot manufacture emit.

**Why the asymmetry is structural.** Routing is **one relational rule** that fires regardless of which noun it is, so it generalizes. EMIT needs **each token's own output coordinate**, learnable only by observing that token as a target — there is no general rule and no structural bridge from "the identity that was copied" to "the row of `lm_head` that emits it." The entire readout is a single learned linear map (`logits = lm_head(final_norm(x))`) with the head **untied** from the embedding; nothing derives an unseen token's output direction from the copied payload. Unseen tokens cannot be emitted regardless of vocabulary size.

### 5.5 Synthesis: binding = generalizable COPY + non-generalizable EMIT

Across every arm, the root cause is one thing: **the substrate has no content-agnostic, token-identity-preserving COPY/MOVE primitive** — the one operation attention provides natively and binding fundamentally requires. Every substrate operator (chaos routing, ring-mean lateral mixing, gated memory blend, per-hop RMSNorm, time mean-pool) is a **mixing / averaging / smoothing** operator that destroys the per-token identity binding must preserve and relocate. The only thing that produced binding at all was bolting on literal attention and *supervising* its query-key; and because emit is irreducibly per-token, even that generalized only on the copy half. The clean final statement:

> **Binding = a COPY step (routing) that generalizes + an EMIT step (identity readout) that does not.** The architecture can be taught to *route* generally, but with an averaging substrate and an untied linear readout it cannot be taught to *emit* generally.

The structural fix is not more data or steps: make EMIT inherit routing's generality with a **pointer/copy readout** — output `P(token) ∝ similarity(token_embedding, copied_source_representation)`, i.e. *emit the thing that was copied*, bypassing the learned value/output transforms and the untied head — together with **tying `lm_head` to the embedding**. Only then does emit follow routing and generalize to unseen entities.

---

## 6. Conclusion

Spider Web demonstrates that Lorenz-63 chaotic dynamics can serve as a stable, trainable routing mechanism without divergence, and that a constant-size Solar Ring Memory yields a fixed-size sequential context store. It produces locally fluent text at 3.48M parameters. But its defining result is a negative one made precise: **it cannot bind**, and the reason is architectural rather than a matter of loss, storage capacity, scale, or training budget.

The four-link decomposition and the disjoint-vocabulary study localize the failure to a single missing primitive — a **content-agnostic, identity-preserving copy** — and show that this primitive is required at **two** stages: at **routing** (select and move the specific earlier entity to the recall site) and at **output** (emit the copied identity, generalizing to tokens never seen as targets). Attention supplies the first natively via content-addressed selection; a tied or pointer-style readout supplies the second. An architecture assembled from mixing, averaging, and normalization operators with an untied linear head structurally lacks both: the copy half can be supervised into generalizing, the emit half cannot be coaxed into it at all.

The contribution is methodological as much as architectural: by *building* the non-attention alternative and instrumenting exactly where it breaks, we give a constructive, measured account of *what attention is for*. Attention is not merely a convenient way to compute context; in the binding regime it is the carrier of an identity-preserving copy that averaging-based substrates cannot reconstruct. The path to general binding in this architecture is therefore specific and testable: add a real copy primitive at the routing stage and a pointer/tied readout at the output stage — not a sixth retrieval heuristic, and not more parameters or data.

---

## 7. Limitations and honest negative results

- **Not a competitive LM.** Eval CE 4.77 is a 20,000-step single-GPU proof-of-concept on a 5K vocabulary; no valid apples-to-apples comparison to published TinyStories baselines exists, and we claim none.
- **Binding is unsolved.** No arm achieved *general* binding. Retrieval supervision achieved only **in-template, in-vocabulary** binding (60% top-5) and **0% on unseen entity types**. The fixed-lag recall, orbital query-read, and gated hybrid-lookback arms each produced **no** real binding (0/4 probes; A/B null).
- **The proposed copy/pointer readout is a hypothesis, not a result.** It is motivated by the decomposition but has **not** been implemented or measured here.
- **Scaling advantage unrealized.** SRM's O(1) memory claim does **not** manifest within the tested range against a parameter-matched Transformer (7.8× vs 7.3× growth over 32× sequence length), and the serial hop loop is **65–149× slower** in throughput.
- **Router ablation incomplete.** The Lorenz-vs-linear router ablation was interrupted; the preliminary signal had the **linear router slightly ahead**, so chaos routing is not yet shown to help.
- **Method caveats made explicit.** CE is near-neutral across binding arms and is *not* the decisive metric — binding/linear probes are. Early small-n probes (4 prompts) were misleading (gave 4/4 then 3/4); statistical tests use n ≥ 120. One control-arm "ball" hit was rank-5 frequency noise (it also surfaced for the wrong entity), not binding. Some structural-read A/B runs were warm restarts (model weights only; optimizer/scheduler state not saved), not exact resumes.

---

## References

*Placeholder — to be filled with real citations. Anticipated topics: TinyStories (Eldan & Li); RoPE; Lorenz-63 dynamical systems; binding/variable-binding in neural networks; coreference/anaphora benchmarks; pointer/copy networks; weight tying for output softmaxes; mechanistic-interpretability induction/copy heads. No citations are fabricated here.*
