# Milestone 2 — Question-dependent visual reasoning

## Status

**Protocol fixed; experiment pending.**

The first baseline budget and validation gates were agreed before research
training. The relational training, checkpoint, and control infrastructure has CPU
correctness coverage. Relational CUDA validation and the research run remain to
be performed by the user on Kaggle. No research results are recorded here yet.

The capability question is whether a model trained from scratch can answer varied
one-hop questions about three-object scenes on held-out geometries, with evidence
that both images and questions matter. A recurrence advantage is not required.

## Fixed experiment specification

The corpus contains horizontal rows of three objects with distinct shapes and
colors, using squares, circles, triangles, and red/green/blue/yellow. Each image
has four valid questions of the form “What color is the object immediately left
of the square?” or the corresponding right/other-shape question. Answers are
single color tokens. Labels and rendering are independently tested; scene/query
metadata is supervision and inspection information, never model input.

Corpus configuration: seed **0**, image size **32**, object sizes **(6, 8)**,
and geometry counts **16/4/4**. Each geometry expands into all 144 shape/color
variants. Geometry assignments are disjoint across splits; colors and shapes
are shared. Stored manifest order is authoritative.

| Split | Geometries | Images | QA examples |
| --- | ---: | ---: | ---: |
| Training | 16 | 2,304 | 9,216 |
| Validation | 4 | 576 | 2,304 |
| Test, reserved for later | 4 | 576 | 2,304 |

Use [the explicit model configuration](../../configs/relational_baseline.yaml):
vocabulary 17, sequence capacity 128, width 64, four attention heads, feed-forward
width 256, and one block each in the prelude, weight-shared recurrent core, and
coda. Recurrence depth is **R=2** throughout training and evaluation. Images have
three channels and patch size 8; dropout is 0 and LayerNorm epsilon is `1.0e-5`.
This is the existing architecture with direct patch embeddings and no pretrained
components or warm start from Milestone 1.

Training settings are fixed as follows:

| Setting | Value |
| --- | --- |
| Initialization/training seed | 0 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.001 / 0 |
| Betas / epsilon | (0.9, 0.999) / 1e-8 |
| `foreach` / `fused` | False / False |
| Execution | Single process, single CUDA device (`cuda:0`), float32 |
| Budget | 10 complete epochs |
| Total updates | 2,880 (288 per epoch) |
| Total QA presentations | 92,160 |
| Evaluated checkpoint | Final epoch-10 `last.pt` |

Train with answer-only shifted cross-entropy and the existing prefix mask. Use
the existing deterministic epoch shuffle with no dropped examples. No scheduler,
early stopping, automatic extension, or selection of an earlier checkpoint is
part of this baseline. This is nine times the update count of Milestone 1, not
a matched-compute comparison.

Evaluate ordinary validation metrics at epoch zero and after every epoch. At
epoch 10, evaluate all validation controls at batch size 32 and R=2: correct and
blank images once each, plus ordinary image and within-image question shuffles
using seeds **0, 1, 2, 3, 4**. Retain all seeds and natural coincidences. Use the
same frozen checkpoint in every condition. Evaluation forwards only images and
question tokens, and full-vocabulary argmax counts non-color outputs as errors.

## Validation acceptance gates

**Every row must pass**, using unrounded report values. Thresholds are inclusive.
The three control gaps compare overall QA accuracy; 30 percentage points means
a difference of at least **0.30**, not a 30% relative improvement.

| Measure | Required value |
| --- | ---: |
| Correct-input overall accuracy | ≥0.90 |
| Correct-input accuracy for each of the six question texts | ≥0.80 each |
| Correct-input all-four accuracy | ≥0.80 |
| Correct accuracy minus blank-image accuracy | ≥0.30 |
| Correct accuracy minus mean shuffled-image accuracy (all five seeds) | ≥0.30 |
| Correct accuracy minus mean shuffled-question accuracy (all five seeds) | ≥0.30 |

Read overall and per-question metrics from `results.correct`, all-four accuracy
from `results.correct.all_four.accuracy`, and the three differences from
`results.accuracy_gaps` in `controls.json`. Compare the JSON values, not rounded
terminal percentages. Missing/nonfinite metrics or an incomplete/wrong-budget
run are not a pass.

Also report different-answer-pair accuracy, invalid predictions, cross-entropy,
individual shuffle metrics, shuffle summaries, and question-shuffle answer-match
fractions. These are diagnostics, without additional acceptance thresholds.

All-four accuracy requires every question for an image to be correct. There are
five unordered question pairs per image with different reference answers;
different-answer-pair accuracy requires both predictions to be correct. An
image-only middle-color predictor can score 50% overall but zero on both grouped
metrics. A perfect model has expected accuracy 37.5% under ordinary within-image
question shuffling because some replacement questions share the original answer.
Report each seed's realized answer-match fraction; do not demand 25% for this
control. See [control definitions and usage](../relational_training.md).

These are practical capability gates, not statistical significance tests. The
four validation geometries and their question/appearance variants are correlated;
five shuffle seeds are not five independent training runs. This baseline provides
scoped synthetic-task evidence, not a claim about broad intelligence, persistent
memory, real-image reasoning, or recurrence superiority.

## Decision and evidence handling

If any gate fails, record the full baseline and diagnose it before proposing a
new experiment. Preserve this protocol, its thresholds, and its artifacts;
document any later protocol separately instead of changing the original gates
after seeing results. Do not extend training or select an earlier epoch to make
this run pass. Resume an interrupted run only to reach the original ten-epoch
budget; `--epochs` specifies additional epochs, not the desired total.

If all gates pass, the next step is a separately authorized frozen-checkpoint
evaluation on the reserved test split. Do not infer on that split during baseline
training or validation controls. Passing validation alone does not close the
milestone. The frozen-test protocol and close-out remain subsequent steps.

Keep smoke checks separate from research artifacts and start the baseline from
scratch. Smoke accuracy is not a basis for tuning this baseline. For a hardware
failure, preserve logs and resume from the last complete epoch on the saved
backend. Record any change of hardware/software; CPU bitwise-resume tests do not
promise exact replay across different CUDA environments.

At run completion, retain the actual code revision, exact model YAML, full console
logs, selected GPU and runtime information, source/generated manifest and hash,
`settings.json`, `metrics.json`, the embedded-manifest copy, final `last.pt`, and
validation `controls.json` with its checkpoint hash. Record these identifiers and
the pass/fail assessment here when results arrive. Manifest generation includes
software versions, so byte hashes can differ between environments even when
scene records agree; use the actual manifest embedded in the run checkpoint.

[Kaggle commands and resume instructions](../relational_training.md) ·
[Research overview](../../README.md)
