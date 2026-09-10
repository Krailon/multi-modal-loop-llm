# Milestone 2 — Training geometry diversity

## Status and question

**Protocol fixed; implementation ready for Kaggle. No research results yet.**

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

Preserve both prior runs and their protocols. Return the new archive for review
before choosing any further data, budget, or architecture change.
