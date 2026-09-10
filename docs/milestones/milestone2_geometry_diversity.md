# Milestone 2 — Training geometry diversity

## Status and question

**Run complete; three of nine criteria passed. See [results](#results).**

The [first direct-grounding experiment](milestone2_shape_grounding.md#results)
reached 94.21% training accuracy but only 56.25% validation accuracy. This follow-up
tests broader training geometry coverage using the same direct questions,
renderer, architecture, initialization seed, and training budget. It does not
change the image frontend or establish a recurrence advantage.

## Fixed corpus derivation

The sole input is the completed `milestone2_shape_grounding_artifacts.zip`, or its
extracted directory. The reference checkpoint and reports are verified against
recorded hashes. Its exact embedded source manifest remains authoritative:
`c5a9102eb619bfb134af48177bc1d1c2a083ce9b98d18a116be817b218c78b6c`.
Reference checkpoint:
`529cd69edb893236d23ff0e9b704a15f5a0d8e3db0486d988f942f08f5ccb41d`.

Retain the original 16 training geometries and add 112 legal geometries. Exclude
all original validation/test geometries. Keep both held-out splits' records and
ordering exactly unchanged, including their original relational metadata. No test
examples are constructed or evaluated. New direct questions remain “What color
is the square/circle/triangle?” in that order for each stored image.

Selection version 1:

1. Enumerate the existing canonical geometry catalog. Exclude held-out geometries
   and size triples absent from the original training set.
2. Multiply each original ordered size-triple count by eight to obtain quotas.
3. Shuffle eligible catalog entries using local `random.Random(0)`.
4. Start with the original 16 geometries. Choose each additional geometry by most
   newly covered absolute-coordinate features, then most newly covered joint
   patch-offset features, breaking ties by shuffled catalog order. Respect quotas.
5. Sort selected geometries lexicographically, expand all shape permutations and
   distinct color permutations in the established order, then shuffle records with
   `random.Random("0:train")`.

An absolute-coordinate feature is `(object position, size, axis, coordinate)`.
A joint patch-offset feature is `(object position, size, left % 8, top % 8)`.
Object positions are the left, middle and right slots. Coverage is measured over
the eligible catalog, not chosen from validation predictions or error cases.

| Coverage | Original | Expanded | Attainable |
| --- | ---: | ---: | ---: |
| Absolute-coordinate features | 70 | 204 | 204 |
| Joint x/y patch-offset features | 43 | 329 | 384 |

This is complete marginal coordinate coverage, not complete coverage of joint
scenes or every patch-offset combination. Selection intentionally prioritizes
coverage over uniform frequency. Save covered/missing features and size counts.
The ordered size-triple quotas are:

| Sizes, left to right | Original | Expanded |
| --- | ---: | ---: |
| 6,6,6 | 4 | 32 |
| 6,6,8 | 2 | 16 |
| 6,8,6 | 3 | 24 |
| 8,6,6 | 1 | 8 |
| 8,6,8 | 2 | 16 |
| 8,8,6 | 3 | 24 |
| 8,8,8 | 1 | 8 |

Each geometry retains all 144 shape/color variants and three direct questions.
All colors remain balanced conditional on geometry and queried shape. The derived
manifest uses the existing validated format; its config alone is not a recipe
for regenerating it. `derivation.json` records the source content/hash, derivation
version, selection settings, geometry list, coverage, and derived manifest hash.

## Training and evaluation

| Setting | Reference | Follow-up |
| --- | ---: | ---: |
| Training geometries | 16 | 128 |
| Training images | 2,304 | 18,432 |
| Training QAs | 6,912 | 55,296 |
| Updates | 2,880 | 2,880 |
| QA presentations | 92,160 | 92,160 |
| Mean presentations per QA | 13⅓ | 1⅔ |
| Full passes and additional batches | 13 + 72 | 1 + 1,152 |

All 55,296 QAs are visited: 18,432 once and 36,864 twice. Previously, 4,608 QAs
were visited 13 times and 2,304 were visited 14 times. Preserve deterministic
per-pass shuffling with `random.Random(f"0:train:{epoch}")`, batch size 32, and
record exact exposure totals by geometry, shape, color and queried-object size.

Use fresh seed-0 weights and the unchanged `configs/relational_baseline.yaml`:
172,928 parameters, 32×32 RGB, patch size 8, width 64, four heads, one block each
in prelude/core/coda, R=2, dropout 0. Vocabulary and six-token direct questions
are unchanged. Retain float32, single CUDA device, AdamW lr 0.001, weight decay 0,
betas (0.9,0.999), eps 1e-8, `foreach=False`, `fused=False`.

Validate at steps 0, 288, …, 2,880. Select the final checkpoint irrespective of
validation history. Do not extend training, select an earlier checkpoint, sweep
recurrence, or infer on test. Training resume remains outside this increment.
The two direct-grounding runs match sequence lengths, architecture, update count
and training QA presentations; the expanded final training diagnosis costs more.

Evaluate all training and unchanged validation QAs. Reuse per-shape, geometry,
position, size, all-three, circle/square pair, identical-prediction and position
confusion diagnostics. Separately report original-geometry and added-geometry
training subsets (6,912 and 48,384 QAs). Save every prediction and the error preview.
The source manifest identifies these subsets; model inputs contain no metadata.

Keep blank-image, shuffled-image and shuffled-question validation controls with
seeds 0–4. Preserve recipient labels and sequence layout. Keep the same nine
criteria: at least 95% training accuracy for each shape, at least 90% validation
accuracy for each shape, and three overall control gaps of at least 30 percentage
points. Assess these on the full new training corpus and original validation set;
subset metrics are diagnostic. These do not replace Milestone 2's relational gates.

`comparison.json` reports current-minus-reference validation differences, including
per-shape accuracy, loss, grouped accuracy and control gaps. Compare training on
the original-layout subset separately from the enlarged corpus. Lower loss is
better; a small validation difference on four geometries is limited evidence.

The intervention changes geometry coverage and repetition frequency together.
Improvement supports this data intervention without isolating those two effects.
Failure at the fixed budget could reflect insufficient learning of the larger
corpus; it would not by itself prove an image-frontend limitation.

## Kaggle and local interfaces

Import [the notebook](../../notebooks/kaggle_milestone2_geometry_diversity.ipynb)
from GitHub after committing/pushing these changes. Enable GPU and internet. Attach
the completed **shape-grounding** archive, set `REFERENCE_SOURCE`, select a
committed `REPO_REF`, and run all cells. Use a fresh output directory. Reference
weights are audited but never used to initialize or train the new model.

The archive `milestone2_geometry_diversity_artifacts.zip` contains:

- `data/`: original/derived manifests, derivation, coverage and presentation reports.
- `training/`: self-contained checkpoint, settings/history and actual-budget exposure report.
- `diagnosis/`: train/validation predictions, summary, controls, comparison and HTML preview.
- `provenance/`: protocol, reference reports, revision/runtime and final hashes.
- `logs/`: generation, training and evaluation output.

Staged reference weights are excluded. The generation command is
`scripts/generate_geometry_diversity.py --source-manifest … --output-dir …`.
Its research default is 128 geometries; `--smoke --geometry-count …` supports small
infrastructure checks. The trainer adds `--geometry-source-manifest …` alongside
`--manifest …`, reproducing the derivation before training. Without that argument,
the existing strict original-manifest research path remains available. Frozen
checkpoint loading verifies embedded derivation provenance; older checkpoints
without that field remain supported. Existing evaluation commands are unchanged.

Preserve both prior runs and this completed run with their original protocols.
The audited results and interpretation follow.

## Results

**Run complete: three of nine criteria passed; all three dependence controls pass.**
The six per-shape training/validation accuracy criteria remain unmet. Milestone 2
is incomplete and the package version is unchanged.

### Provenance and audit

Kaggle ran a clean checkout at `54010afe4d636ae85cff3c60117e5cebbc77e74f`,
using PyTorch 2.10.0+cu128, CUDA 12.8,
a Tesla T4 and two CPU threads. The final checkpoint completed exactly 2,880
updates and 92,160 QA presentations at R=2: one complete pass and 1,152 batches of
the second pass. Neither model selection nor extra training was performed.

The local artifact audit verified checkpoint/report hashes, settings and optimizer
step counters, the recorded history, and prediction-derived summaries, subsets,
criteria and comparisons. The corpus derivation was independently reproduced:
128 training geometries, coverage of 204/204 coordinate features and 329/384 joint
patch-offset features, with the original size-triple proportions. The exact
validation/test records and ordering remain unchanged. The deterministic stream
visits 18,432 QAs once and 36,864 twice. No test inference was performed.

Retain `milestone2_geometry_diversity_artifacts.zip` with these identities:

- Archive: `3ca3ad2283ec47a02e79597d64bd9e7cc2cf39a3219fbd15dc6a26b462bce4ec`
- Derived manifest: `047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975`
- Checkpoint: `49012c1c488478c7eb66e27b979b59c22a47a92d696120ad9279718afbd71456`
- Summary: `5351a33268a02aa743a138c4849ba90c12ee4edd3a39e5479dbc4ef851528d88`
- Controls: `bd0a025992218418eec1a1ef23ebd8363f38165f53cb20a2aed91de67c4e872e`
- Comparison: `d64f55192bc1ef6aefbf354758c05489fcace1d86094d7c006a548ffae4eaa05`

### Final checkpoint and unchanged criteria

| Metric | Training | Validation |
| --- | ---: | ---: |
| Correct / total | 38,785 / 55,296 | 1,155 / 1,728 |
| Overall accuracy | 70.14% | 66.84% |
| All-three accuracy | 35.41% | 31.25% |
| Circle/square pair accuracy | 37.55% | 32.29% |
| Loss | 0.5651 | 0.6160 |

| Criterion | Result | Required | Outcome |
| --- | ---: | ---: | --- |
| Training: circle | 56.77% | ≥95% | Fail |
| Training: square | 62.07% | ≥95% | Fail |
| Training: triangle | 91.59% | ≥95% | Fail |
| Validation: circle | 52.08% | ≥90% | Fail |
| Validation: square | 59.03% | ≥90% | Fail |
| Validation: triangle | 89.41% | ≥90% | Fail |
| Correct minus blank images | 41.84 pp | ≥30 pp | Pass |
| Correct minus mean shuffled images | 42.27 pp | ≥30 pp | Pass |
| Correct minus mean shuffled questions | 33.92 pp | ≥30 pp | Pass |

Values above are rounded for display; outcomes use the unrounded report values.
The assessment uses the full expanded training corpus. Validation controls score
25.00% with blank images, 24.57% with mean shuffled images, and 32.92% with mean
shuffled questions. The same three-of-nine count as the reference conceals a
change: the reference passed triangle training and two image controls; this run
passes all three dependence controls and no per-shape accuracy criteria.

### Comparison with the 16-layout reference

| Validation measure | 16 layouts | 128 layouts |
| --- | ---: | ---: |
| Overall accuracy | 56.25% | 66.84% |
| Circle accuracy | 43.40% | 52.08% |
| Square accuracy | 58.33% | 59.03% |
| Triangle accuracy | 67.01% | 89.41% |
| All-three accuracy | 24.13% | 31.25% |
| Circle/square pair accuracy | 31.08% | 32.29% |
| Loss | 1.7399 | 0.6160 |

Overall validation accuracy rose by **10.59 percentage points**, with improvement
on each of the four unchanged validation geometries:

| Validation geometry | 16-layout reference | 128-layout follow-up |
| --- | ---: | ---: |
| `validation:g000` | 59.72% | 71.99% |
| `validation:g001` | 53.47% | 67.59% |
| `validation:g002` | 62.27% | 72.92% |
| `validation:g003` | 49.54% | 54.86% |

The full training accuracy of 70.14% cannot be directly compared as a measure of
fitting the same corpus with the reference's 94.21%; the corpus is eight times
larger. The shared original-layout subset makes the change explicit:

| Training subset in this run | Correct / total | Accuracy | Loss |
| --- | ---: | ---: | ---: |
| Original 16 layouts | 4,976 / 6,912 | 71.99% | 0.5310 |
| Added 112 layouts | 33,809 / 48,384 | 69.88% | 0.5700 |

Accuracy on the original layouts also fell, from 94.21% to 71.99%. Thus the smaller
training/validation gap partly reflects reduced training-set fit, rather than
mastery of the task. Mean presentations per QA fell from 13⅓ to 1⅔. Geometry
coverage and repetition frequency changed together at fixed update/presentation
budget; this experiment does not isolate their individual effects.

### Remaining errors and learning trajectory

Circle/square swaps account for 453 of 573 validation errors (79.06%). Correctly
answering both questions improved only from 31.08% to 32.29%; identical predictions
for these different-target questions rose from 24.48% to 43.92%. Triangle validation
accuracy improved most, from 67.01% to 89.41%, but remains below the 90% criterion.
The model learned a more transferable partial solution, not reliable direct grounding.

High-confidence validation errors declined sharply: 10 errors have at least 90%
confidence, versus 374 in the reference. Mean confidence on wrong answers fell
from 83.30% to 57.15%, alongside the lower loss and improved accuracy.

Validation reached 70.78% at step 2,592, then fell to **66.84% at the official final
step 2,880**. The earlier monitoring checkpoint is not substituted for the final
result. Recent training-interval losses continued declining, but validation
progress was noisy and does not guarantee gains from further optimization.

### Interpretation and next decision

Broader geometry coverage with reduced repetition improved held-out performance
using the unchanged architecture. Residual training errors and circle/square
confusion leave the learning mechanism unresolved. This is one seed on four
validation layouts, not evidence of broad generalization or a recurrence advantage.

A separately specified, modest increase in training budget on this corpus is a
recommended next controlled question. At the close of this review, no new budget
or experiment had been established; further training required a separate protocol. Preserve this
run and its original protocol before considering further data, optimization or
image-frontend changes.

The separate [training-budget follow-up](milestone2_training_budget.md) now fixes
a fresh 5,760-update run on this exact corpus. Its [completed results](milestone2_training_budget.md#results) show improved
circle/square grounding and five of nine criteria passing. This geometry-diversity
experiment and its original budget remain unchanged.
