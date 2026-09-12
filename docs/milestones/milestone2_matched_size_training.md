# Milestone 2 — Matched size-variant training

## Status and experimental question

**Completed; negative result at the fixed update budget.** Validation accuracy
fell and size-dependent confusion persists; see [results](#results). The
[size intervention](milestone2_size_intervention.md#results) demonstrated strong
size-dependent answers. Does systematic size variation within matched training
arrangements improve circle/square grounding and invariance at the same update
budget? Preserve the architecture and completed experiments as references.

## Fixed corpus and rendering policy

Start from the audited 128-layout manifest in the completed training-budget
archive, SHA256
`047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975`.
Keep all original training images and their order. For each source scene, attempt
circle/square sizes (6,6), (6,8), (8,6), (8,8), preserving object origins, identities,
colors and triangle size. Add the family only if **all four** variants satisfy:

- One-pixel canvas margins and at least one blank column between adjacent boxes.
- Complete visibility, distinct shape/color identities, unchanged object origins.
- Vertical center alignment may change; this is the only relaxed rendering rule.

Reuse existing shape masks and rendering primitives. Original relational scene
validation remains unchanged; the new corpus has its own manifest and dataset.
Deduplicate by complete ordered scene identity (shape, color, left, top, size).
Preserve original order, then append new scenes sorted by that identity. Quartets
share scene entries when deduplication finds an existing original.

| Corpus measure | Count |
| --- | ---: |
| Original images retained | 18,432 |
| Eligible complete families | 7,008 |
| Additional unique images | 21,024 |
| Total training images | 39,456 |
| Total training QAs | 118,368 |
| Original training QAs | 55,296 |
| Added training QAs | 63,072 |

The preflight derived manifest SHA256 is
`25071dc462e1e8bdadee82ca7b0531cdde871c4f9735613532f41210c9192828`.
Derivation records every eligibility decision, exclusion reason and family's four
indices in the unique union. Preserve exact validation/test records and ordering
inside the original source manifest. Reject any original or generated training
scene whose ordered origin coordinates match a held-out layout, regardless of
shape, color or size. The local preflight found no such overlaps.

## Fixed training procedure

Use fresh seed-0 weights, the unchanged `configs/relational_baseline.yaml`
(172,928 parameters, 32×32 RGB, patch size 8, width 64, four heads, one block each
in prelude/core/coda, R=2, dropout 0), float32, single CUDA device, batch size 32,
and direct six-token questions. Retain AdamW lr 0.001, weight decay 0,
betas (0.9,0.999), eps 1e-8, `foreach=False`, `fused=False`, and answer-only loss.

Train for **5,760 updates / 184,320 QA presentations**, validating at steps
0, 288, …, 5,760. Shuffle uniformly per pass with
`random.Random(f"0:train:{epoch}")`. No family batching, mixture weighting,
consistency loss, resume, early stopping, schedule change or automatic extension.
Evaluate the final checkpoint irrespective of the validation trajectory.

The budget is one full pass of 3,699 batches plus 2,061 batches of pass two.
52,416 QAs appear once and 65,952 appear twice, averaging about 1.56 presentations,
compared with 3⅓ on the reference corpus. Recompute exact exposures by geometry,
queried shape, color and size. The intervention changes data coverage, alignment
distribution and repetition frequency together; it does not isolate their effects.
Training updates match, while expanded full-corpus diagnostics add evaluation cost.

## Evaluation and comparisons

Evaluate the final frozen model on the full expanded training set and unchanged
1,728 validation QAs. Keep original and added training subsets explicit. Reuse
per-shape, size, position, geometry, circle/square pair, identical-prediction,
all-three and confidence diagnostics, and the original blank/shuffled-image and
within-image shuffled-question controls with seeds 0–4. Record full predictions
and relative-size swap classifications, including the difficult validation layout.

Retain the nine direct-grounding criteria: ≥95% per-shape accuracy on the **full
expanded training corpus**, ≥90% per-shape validation accuracy, and ≥30-point
validation gaps against all three dependence controls. These remain separate
from the relational milestone-completion gates.

Compare with the completed 5,760-update training-budget checkpoint, SHA256
`2069297aabdff24489bf2be5ef2f0c5a53989f4daaea312cd55e51a72282f8bc`.
Training-fit comparisons use the identical original 55,296-QA subset; expanded
training accuracy is a separate measure. Compare validation accuracy, per-shape
and grouped metrics, loss and control gaps without changing denominators.

Repeat the established five-condition size intervention on the **same 384
eligible validation images**, including its original control: 5,760 additional
QA evaluations. Compare per-condition accuracy, pair accuracy, invariance and
reversal behavior with the completed size-intervention report. Training retains
margins; this diagnostic still permits edge contact and retains its original
limitations. No new size-intervention pass/fail thresholds are introduced.

Improvements in smaller-square cases, paired correctness and correct invariance
would support this data intervention. Aggregate improvement alone is insufficient.
Failure would not uniquely establish a frontend or architecture limitation.
This remains one seed and four validation layouts, with no recurrence comparison.
No test inference is authorized. Milestone 2 remains incomplete.

## Interfaces and Kaggle workflow

Use [the notebook](../../notebooks/kaggle_milestone2_matched_size_training.ipynb)
after committing/pushing. Enable GPU and internet; select a committed `REPO_REF`
(commit SHA preferred) and fresh checkout/output directories. Attach both:

- `milestone2_training_budget_artifacts.zip` as `REFERENCE_SOURCE`.
- `milestone2_size_intervention_artifacts.zip` as `INTERVENTION_SOURCE`.

ZIPs and extracted run directories are accepted. Both references are audited;
reference weights never initialize training. The second archive supplies already
recorded intervention metrics, avoiding additional reference-model inference.
Kaggle's installed PyTorch is preserved; the notebook uses `cuda:0`.

`prepare_matched_size(repo, root, source, intervention_source)` derives the corpus
and records the protocol. `run_matched_size(run)` invokes separate training and
evaluation entry points; `archive_matched_size(run)` packages the result. Helpers
live in `multimodal_loop.eval.kaggle_matched_size`. The corpus and checkpoint use
the distinct `matched_size_color` kind. Existing formats and entry points retain
their behavior; frozen loading does not provide training resume.

Local command interfaces are `scripts/train_matched_size.py` (`--manifest`,
`--model-config`, `--output-dir`, `--device`, `--max-steps`, `--evaluation-interval`)
and `scripts/evaluate_matched_size.py` (`--checkpoint`, `--output-dir`, `--device`,
`--shuffle-seeds`). A small non-reference training fixture requires `--smoke` and
is marked accordingly. Research settings are enforced by the notebook workflow.

The output `milestone2_matched_size_training_artifacts.zip` contains data and
family derivation, the fresh checkpoint and history, exposure reports, complete
predictions, original/added comparisons, size-intervention results, HTML previews,
reference reports, runtime/revision provenance, hashes and logs. Staged reference
weights are excluded. The completed archive was reviewed as recorded below.


## Results

The completed `milestone2_matched_size_training_artifacts.zip` follows the fixed
5,760-update / 184,320-presentation protocol. This is a negative result for the
specified intervention and budget: validation accuracy regressed, and the matched
resizing diagnostic shows little improvement in correct size invariance.

### Provenance and audit

The archive reports clean revision
`08ba56756d1f5d4d47d9518d8812e844e5c284dd`, PyTorch 2.10.0+cu128, CUDA 12.8,
one Tesla T4, float32 and two CPU threads. Model configuration, R=2, seed-0
training settings, AdamW settings and the final budget match the protocol.

Local review verified all 11 hashes in `provenance/final.json`, regenerated the
expected corpus, checked checkpoint configuration, history and progress, and
recomputed deterministic exposure counts. All 118,368 training and 1,728
validation prediction records were checked against their scene labels and report
summaries. The size-intervention metrics were recomputed from 5,760 saved
predictions. No new model inference or test inference was performed during review.

- Manifest SHA256: `25071dc462e1e8bdadee82ca7b0531cdde871c4f9735613532f41210c9192828`
- Final checkpoint SHA256: `2bf8fcaf0e9cb0f2474555e2a795b8aca8db776fef8cd1478cf8836f553bd551`
- Summary SHA256: `dcaea79f40a087979a3a24537f151fbdf310cac1225bd58210a54651c031289a`
- Controls SHA256: `260967ed9215d8706d30be9010b9a9197c3caa3f04855959ada8bd8dc1cff3c7`
- Comparison SHA256: `98389fdc8f6874b772b115ad019187ae708eb91b1d27d09f0d1462bd3791e736`
- Size-intervention summary SHA256: `73acec2d8f1696a9cfe227fa911344925b063e8e131a61e9884e259cc7a8fa8b`

The corpus retains all 18,432 originals and adds 21,024 unique images from 7,008
eligible families, totaling 39,456 images. Margins and fixed origins are retained;
vertical alignment alone is relaxed. Held-out records and ordering remain
unchanged, with training origins separated from validation/test origins. Exact
exposure counts match the protocol: 52,416 QAs seen once and 65,952 twice, averaging
1.56 presentations versus 3.33 in the reference run.

### Final checkpoint and reference comparison

The reference is the completed [5,760-update training-budget run](milestone2_training_budget.md#results).
Training comparisons below use the identical original 55,296 QAs. The expanded
training corpus is reported separately to avoid changing the comparison denominator.

| Measure | Reference | Matched-size training |
| --- | ---: | ---: |
| Original training subset accuracy | 77.51% | 77.39% |
| Original training subset loss | 0.4351 | 0.4461 |
| Original training circle/square pair accuracy | 59.28% | 58.36% |
| Validation accuracy | 74.71% | 70.08% |
| Validation loss | 0.5598 | 0.5692 |
| Validation circle accuracy | 65.28% | 58.51% |
| Validation square accuracy | 66.32% | 62.67% |
| Validation triangle accuracy | 92.53% | 89.06% |
| Validation circle/square pair accuracy | 56.77% | 49.65% |
| Validation all-three accuracy | 56.08% | 45.66% |
| Validation identical circle/square predictions | 15.97% | 20.31% |

Expanded training accuracy is 92,605/118,368 = 78.23%, with loss 0.4248;
added-subset accuracy is 78.97% on 63,072 QAs. Validation accuracy is
1,211/1,728 = 70.08%, down 4.63 percentage points. There are no invalid predictions
in either full training or validation diagnostics.

| Criterion | Result | Required | Outcome |
| --- | ---: | ---: | --- |
| `train_circle` | 68.34% | ≥95% | Fail |
| `train_square` | 68.90% | ≥95% | Fail |
| `train_triangle` | 97.46% | ≥95% | Pass |
| `validation_circle` | 58.51% | ≥90% | Fail |
| `validation_square` | 62.67% | ≥90% | Fail |
| `validation_triangle` | 89.06% | ≥90% | Fail |
| `correct_minus_blank` | 45.08 pp | ≥30 pp | Pass |
| `correct_minus_shuffled_images_mean` | 45.78 pp | ≥30 pp | Pass |
| `correct_minus_shuffled_questions_mean` | 36.74 pp | ≥30 pp | Pass |

Four of nine criteria pass, versus five in the reference. Training criteria use
the full expanded corpus. All three dependence controls still pass: blank-image
accuracy is 25.00%, mean shuffled-image accuracy 24.31%, and mean shuffled-question
accuracy 33.34%. These are recipient-target controls; the results support use of
both inputs, not reliable shape identification. Criteria use unrounded values.

### Relative size and matched resizing

Ordinary validation circle/square pair accuracy remains strongly size-dependent:

| Relative size | Reference | Matched-size training |
| --- | ---: | ---: |
| Square smaller than circle | 24.48% | 9.38% |
| Equal size | 51.04% | 50.00% |
| Square larger than circle | 94.79% | 89.58% |

Each row contains 192 validation images. In the smaller-square group, the new
model answers the square question correctly on 42/192 and the circle question on
39/192; 302 of the 303 errors select the other circle/square object's color.
On the original training subset, smaller-square pair accuracy falls from 15.58%
to 14.46%. On the expanded training set it is 21.09%, versus 92.01% for larger
squares. The failure therefore is not confined to unseen layouts.

The repeated [size intervention](milestone2_size_intervention.md#results) uses the
same 384 eligible validation images and unchanged five conditions. Pair accuracy
requires both circle and square answers to be correct.

| Condition | Reference overall | New overall | Reference pair | New pair |
| --- | ---: | ---: | ---: | ---: |
| Original | 71.53% | 70.14% | 53.12% | 50.26% |
| Circle 6, square 6 | 70.83% | 71.79% | 50.26% | 49.74% |
| Circle 6, square 8 | 92.97% | 92.53% | 88.80% | 90.10% |
| Circle 8, square 6 | 46.44% | 48.61% | 13.28% | 14.06% |
| Circle 8, square 8 | 82.81% | 81.68% | 61.98% | 62.24% |

Both answers remain correct across all four resized conditions on only
48/384 images (12.50%), compared with 44/384 (11.46%). This small descriptive
change does not demonstrate resolution of the size dependence. Diagnostic
variants still permit edge contact, unlike training; the previously documented
rendering and eligibility limitations continue to apply.

The difficult ordinary validation layout `validation:g003` improves slightly
from 56.25% to 57.64%, while accuracy on each of the other three layouts declines.
It remains the weakest layout.

### Trajectory, interpretation and next decision

Validation accuracy peaks at 73.90% at step 5,472 and finishes at 70.08% at step
5,760. The final checkpoint remains the reported result as specified; the peak
does not replace it or authorize further training.

Matched size variation under this budget did not produce reliable grounding or
correct invariance. Coverage, alignment distribution and repetition frequency
changed together, so their effects cannot be isolated. The result does not show
that size variation can never help, nor uniquely identify an image-frontend or
architecture limitation. Evidence is still limited to one seed and four
validation layouts; no recurrence comparison was performed.

A proposed next decision is a small, balanced training-fit sanity check on valid
matched size quartets using the unchanged model. Its purpose would be to test
whether the model can fit the distinction with repeated exposure before another
broad data or architecture change. That experiment is not specified or authorized
by this results record. Milestone 2 remains incomplete, and test inference remains
reserved for a later authorized step. Preserve all protocols and artifacts.
