# Milestone 2 — Convolutional image tokens in the recurrent transformer

## Status and question

**Specified and implemented; research run pending.** The
[standalone CNN](milestone2_cnn_baseline.md#results) solved every training and
transfer question, while the [four-arrangement transformer](milestone2_multi_arrangement.md#results)
reached 44.34% transfer accuracy. Can learned convolutional image tokens improve
position transfer inside the recurrent transformer?

Keep the corpus fixed and change image-token construction. The
[focused literature review](milestone2_cnn_baseline.md#focused-literature-review)
and [Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881)
support investigating this interface, but this experiment is not a reproduction
of that paper. It is an exploratory capability experiment at fixed R=2, not a
recurrence comparison or milestone close-out.

## Architecture and initialization

Replace only the direct image patch projection with:

1. Five full-resolution 3×3 convolutions, channels `3→32→64→64→64→64`;
   stride 1, zero padding 1, bias enabled, ReLU after every layer.
2. 8×8 average pooling with stride 8, giving `[B,64,4,4]`.
3. Row-major spatial flattening to **`[B,16,64]`** visual tokens.

These are the CNN baseline's convolutional layer dimensions, trained afresh.
Regional pooling preserves a spatial grid. Averaging the 16 tokens recovers the
convolutional feature map's global average, before adding positional/modality
embeddings; no global pooling replaces the transformer. Boundaries and pooling
can still introduce position sensitivity; see
[Zhang's shift-invariance study](https://proceedings.mlr.press/v97/zhang19a.html).

Keep the original text, learned absolute position and modality embeddings,
prelude, recurrent core, coda, final LayerNorm and language-model head. Keep all
model dimensions, 16 image tokens, R=2, dropout 0, six-token direct questions,
17-token answer vocabulary, prefix masking and shifted answer-only loss. All
parameters train jointly. Runtime depth overrides and text-only input remain
supported. Scene metadata remains supervision/inspection information only.

The immutable `ConvStemConfig` specifies channels, pool size and stem seed.
`ConvStemEmbedding` and `build_conv_stem_model` live in
`multimodal_loop.model.conv_stem`. The factory constructs the ordinary transformer
on CPU first, then replaces its image embedding under an independent seed-0 CPU
RNG context. The caller seeds the ordinary model with seed 0. The other initial
parameters and the caller's post-construction RNG state match a fresh ordinary
baseline under the same seed/runtime. No trained reference weights are loaded.
Save separate hashes of the initial stem and non-image parameters in settings.

The base `ModelConfig` schema, transformer forward interface and existing
checkpoint formats remain unchanged. Use `configs/relational_baseline.yaml`
with the separate `configs/conv_stem.yaml`. Research settings are enforced;
explicit smoke mode permits small fixtures and widths ending at `d_model`.
Pooling must match the existing patch grid.

## Fixed corpus, exposure and compute

Reuse manifest SHA256
`ce16b936ce4ffcecb231794d9541a3c9d39461fef861c9e02349f9e4d68de8c7`.
Train on `learned/transfer_02/transfer_04/transfer_06`; assess final transfer on
`transfer_01/transfer_03/transfer_05/transfer_07`. Each population has **6,912 QAs,
2,304 images and 576 complete families**. Preserve rendering, all shape/color
combinations, circle/square size quartets, triangle size 8, gaps and margins.
No new geometry or augmentation is introduced.

| Setting | Fixed value |
| --- | ---: |
| Runs / seed | One / 0 |
| Updates / complete passes | 8,640 / 40 |
| Batch size / batches per pass | 32 / 216 |
| Training QA presentations | 276,480 |
| Presentations per QA | Exactly 40 |
| Training-fit monitoring | Step 0 and every 216 updates; 41 records |
| Monitoring QA presentations | 283,392 |
| Final evaluation QA presentations | 13,824 |

Reuse `random.Random(f"0:train:{epoch}")` sampling and the shared direct-grounding
training loop. AdamW: lr 0.001, weight decay 0, betas (0.9,0.999), eps 1e-8,
`foreach=False`, `fused=False`. One CUDA device, float32, two CPU threads.
Monitor training fit only. No resume, early stopping, search, budget extension,
transfer-driven tuning or best-checkpoint selection. Evaluate the final checkpoint.

| Model / forward path | Direct-patch transformer | Convolutional-stem transformer |
| --- | ---: | ---: |
| Parameters | 172,928 | 290,752 |
| Training MACs per QA, 23 positions | 5,014,464 | 137,823,168 |
| Evaluation MACs per QA, 22 positions | 4,793,728 | 137,602,432 |

MAC estimates count convolution, linear and dense attention matrix products,
including masked positions; they exclude pooling, normalization, activations and
backward operations. They are arithmetic estimates, not measured kernel FLOPs.
Record actual training-plus-monitoring time, final assessment time and peak
allocated/reserved CUDA memory separately. Final assessment timing includes
prediction and grouped comparison computation, excluding reference audit/loading
and output writing. Exposure and transformer sequence computation match; **total
compute and parameter count do not**. No speed advantage is assumed.

## Assessment and references

Reuse all 16 per-arrangement training-fit checks: each queried shape ≥99%
(571/576) and correctly invariant circle/square families ≥95% (137/144), without
rounding thresholds. Each training arrangement must pass independently. Correct
invariance requires both answers correct across all four sizes.

Transfer remains descriptive with no new accuracy gates. Report every arrangement,
per-shape accuracy and loss, circle/square pairs, all size conditions, relative-size
swaps, complete-family correctness and confident errors. Compare identical final
transfer examples against both saved references, retaining matched prediction and
correctness changes. The primary reference is the four-arrangement transformer
(44.34%); the CNN (100%) is a second descriptive reference. Do not use the earlier
seven-arrangement aggregate.

Audit both archives' pinned manifests, summaries, predictions and checkpoint
identities without reference-model inference. Reference hashes are in
`eval/cnn_reference.py` and `eval/conv_stem_reference.py`. The evaluated arrangements
are previously examined development populations, not an untouched test split.
Reserved project validation/test inference is unavailable through these entry points.

Improvement would support this image-token construction within the recurrent
model, without isolating convolution from capacity, local pooling or compute.
Weak transfer would leave pooling and the remaining transformer learning setup
unresolved; failed training fit would make transfer interpretation inconclusive.
This single-seed result cannot establish a recurrence advantage or complete
Milestone 2. Broader held-out and relational evaluation require later steps.

## Kaggle workflow, artifacts and checks

Commit/push and import [the notebook](../../notebooks/kaggle_milestone2_conv_stem.ipynb).
Enable GPU/internet and set a committed `REPO_REF`. Both inputs default to the
repository root:

- `TRANSFORMER_SOURCE = REPO_DIR / "milestone2_multi_arrangement_artifacts.zip"`
- `CNN_SOURCE = REPO_DIR / "milestone2_cnn_baseline_artifacts.zip"`

Use fresh checkout/output directories and preserve Kaggle's installed PyTorch.
Helpers `prepare_conv_stem`, `run_conv_stem`, `archive_conv_stem` live in
`multimodal_loop.eval.kaggle_conv_stem`. They invoke `scripts/train_conv_stem.py`
and `scripts/evaluate_conv_stem.py`. Research budgets/configurations are fixed;
`--smoke` labels reduced fixtures/budgets throughout training and evaluation.

The distinct checkpoint kind is `conv_stem_direct_shape_color`, version 1.
Preserve both configurations, model/optimizer state, tokenizer, corpus identity,
history, progress and CPU/device randomness. Frozen loading preserves caller
randomness and supports CPU/CUDA; training resume is unsupported.

Download `milestone2_conv_stem_artifacts.zip`. It includes configurations,
initialization hashes, protocol, manifest, exposure counts, checkpoint, history,
predictions, both reference comparisons, per-arrangement diagnostics, inspection
HTML, runtimes, logs and hashes. The notebook displays all transfer arrangements
alongside both reference models, with correct-family counts and per-shape scores.

Tests cover token order/pooling, matched non-image initialization, parameter/MAC
counts, text-only and multimodal gradients, shared recurrence, leakage prevention
at multiple depths, answer-target alignment, metadata exclusion, checkpoint
reproduction, reference/protocol rejection, exact sampling, CPU CLI smoke runs,
and notebook/archive integrity. Run the full suite with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest` for exact CPU comparisons, and
`ruff check .`. Research results will be recorded here after the Kaggle run.
