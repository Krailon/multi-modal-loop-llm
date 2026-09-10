# Milestone 2 — Question-dependent visual reasoning

## Status

**First baseline and frozen-model diagnosis complete; validation gates not met.**

The original budget and thresholds below were agreed before training and remain
unchanged. The baseline completed on Kaggle CUDA and passed two of six validation
gates. Its results and artifact audit are recorded below. The reserved test split
has not been evaluated as part of this baseline.

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

## First baseline results

**Outcome: validation gates not met; two of six passed.** The CUDA baseline
completed the prescribed 10 epochs / 2,880 updates on a Tesla T4, using Python
3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, and two Torch CPU threads. The user also
reports completing the separate smoke notebook; this archive provides baseline
training and validation evidence, not a separately audited smoke-check archive.

### Training history

| Epoch | Updates | Training loss | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | — | 3.206179 | 0.0000% |
| 1 | 288 | 1.396958 | 1.251384 | 37.1094% |
| 2 | 576 | 0.930189 | 0.883625 | 50.1736% |
| 3 | 864 | 0.665829 | 0.962621 | 50.8681% |
| 4 | 1152 | 0.518065 | 1.100390 | 52.5174% |
| 5 | 1440 | 0.430347 | 1.108956 | 55.9028% |
| 6 | 1728 | 0.365307 | 1.092195 | 56.3802% |
| 7 | 2016 | 0.356425 | 1.330812 | 53.4722% |
| 8 | 2304 | 0.313074 | 1.408357 | 59.3316% |
| 9 | 2592 | 0.306107 | 1.610936 | 57.3351% |
| 10 | 2880 | 0.299399 | 1.724776 | 58.2465% |

Validation loss was lowest at epoch 2 and rose to 1.724776 at epoch 10 while
training loss continued falling. This is consistent with overfitting and
increasingly confident validation mistakes, but does not identify their cause.
No earlier checkpoint was selected and no extra epochs were added. Historical
training loss averages predictions during optimization; final-checkpoint training
accuracy and loss were not measured by this run.

### Frozen validation controls

| Condition | Correct / 2,304 | Accuracy | Cross-entropy |
| --- | ---: | ---: | ---: |
| Correct inputs | 1342 | 58.2465% | 1.724776 |
| Blank images | 576 | 25.0000% | 2.178307 |
| Shuffled images, seed 0 | 577 | 25.0434% | 4.866706 |
| Shuffled images, seed 1 | 616 | 26.7361% | 4.929136 |
| Shuffled images, seed 2 | 575 | 24.9566% | 4.872485 |
| Shuffled images, seed 3 | 580 | 25.1736% | 4.982436 |
| Shuffled images, seed 4 | 602 | 26.1285% | 4.879006 |
| Shuffled questions, seed 0 | 856 | 37.1528% | 3.386835 |
| Shuffled questions, seed 1 | 823 | 35.7205% | 3.507445 |
| Shuffled questions, seed 2 | 871 | 37.8038% | 3.408978 |
| Shuffled questions, seed 3 | 873 | 37.8906% | 3.364596 |
| Shuffled questions, seed 4 | 841 | 36.5017% | 3.459094 |

Mean shuffled-image accuracy: **25.6076%**; mean shuffled-question accuracy:
**37.0139%**. All conditions produced zero invalid color-token predictions.
All-four accuracy was **57/576 = 9.8958%**; different-answer-pair accuracy was
**902/2,880 = 31.3194%**.

| Question anchor | Direction | Correct / 384 | Accuracy |
| --- | --- | ---: | ---: |
| circle | left | 183 | 47.6562% |
| square | left | 197 | 51.3021% |
| triangle | left | 286 | 74.4792% |
| circle | right | 210 | 54.6875% |
| square | right | 196 | 51.0417% |
| triangle | right | 270 | 70.3125% |

| Gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| overall_accuracy | 58.2465% | ≥90% | Fail |
| minimum_question_accuracy | 47.6562% | ≥80% | Fail |
| all_four_accuracy | 9.8958% | ≥80% | Fail |
| correct_minus_blank | 33.2465 points | ≥30 points | Pass |
| correct_minus_shuffled_images_mean | 32.6389 points | ≥30 points | Pass |
| correct_minus_shuffled_questions_mean | 21.2326 points | ≥30 points | Fail |

Images clearly affect performance: both image-control gaps exceed 30 points.
Question shuffling also reduces accuracy, and some different-answer pairs are
answered correctly, demonstrating partial question-dependent behavior beyond
the question-blind middle-color heuristic. Reliability remains inadequate:
all six question texts are below 80%, and fewer than one in ten images has all
four answers correct. The shuffled-question result near 37.5% is not itself
evidence of perfect reasoning; that control retains answer coincidences.

### Artifact audit and provenance

The local review verified checkpoint/manifest/report hashes, exact manifest
copies, embedded and sidecar settings/history, model configuration, finite model
weights, optimizer step counters, and the fixed budget. Regenerated corpus
records match the saved manifest; prescribed image/question permutations and
question-shuffle answer-match fractions were independently recomputed. The
acceptance calculations agree with the stored report. Final ordinary validation
metrics exactly match the correct-input control metrics. No artifact-consistency
problem was found in these checks. This review did not rerun GPU inference.

- Training revision: `f0e1334a4ed582b1aba526eec80506fef2660392`.
- Checkpoint SHA256: `d6e39b3aebc75907bb12f548320f4d77c657c88ee8330c4c2d08940b41cbbffb`.
- Manifest SHA256: `c5a9102eb619bfb134af48177bc1d1c2a083ce9b98d18a116be817b218c78b6c`.
- Controls SHA256: `065fe803b4de56bd61f71ade97ba63fdf9f547d89713fe263642c8b5c2eda4a9`.
- Archive SHA256: `049b45c5fdc964876f17aae7d2edb22941c65b1beadf1f639d53655304a98c79`.

The source is the user-supplied local `milestone2_baseline_artifacts.zip`, whose
root contains `training/`, `validation/`, `data/`, `provenance/`, and `logs/`.
Raw artifacts are not bundled with this Markdown; retain the original archive
separately. The manifest includes generator software versions, explaining why
its hash differs from a corpus generated in another environment.

## Frozen-model diagnosis — completed

The Kaggle archive `milestone2_diagnostics_artifacts.zip` evaluates the unchanged
final checkpoint on all training and validation examples at R=2, batch size 32.
Diagnostic revision: `186266e526d45d063490bfe0e7a2cf93c3b1a3e6`. Runtime: PyTorch
2.10.0+cu128, CUDA 12.8, Tesla T4, two CPU threads. Its checkpoint, manifest, and
original-controls hashes match the baseline. Validation accuracy, counts, invalid
outputs, and loss reproduce the original report exactly. No test inference or
additional training was performed.

| Metric | Training | Validation |
| --- | ---: | ---: |
| Correct / total | 7,641 / 9,216 | 1,342 / 2,304 |
| Overall accuracy | 82.91% | 58.25% |
| Triangle-anchor accuracy | 99.71% | 72.40% |
| Circle-anchor accuracy | 75.75% | 51.17% |
| Square-anchor accuracy | 73.27% | 51.17% |
| All-four accuracy | 35.42% | 9.90% |
| Loss | 0.2862 | 1.7248 |

### Circle/square distinction

A local analysis joined saved predictions to the original scene manifest and
independently checked each target against its anchor's adjacent object. Among
circle/square questions, classify a case as requiring their distinction when
**both** possible anchors have a valid neighbor in the requested direction.
Otherwise the scene boundary allows selecting the anchor even if these two
shape identities are conflated. The questions themselves have unambiguous labels.

| Training category | Correct / total | Accuracy |
| --- | ---: | ---: |
| Circle/square distinction required | 1,547 / 3,072 | 50.36% |
| Circle/square boundary-resolvable | 3,031 / 3,072 | 98.67% |
| Triangle anchor | 3,063 / 3,072 | 99.71% |

For 1,424 / 1,536 (92.71%) same-image, same-direction circle/square question
pairs requiring different answers, the predictions were identical. Both answers
were correct in only 65 pairs. Validation had identical predictions in 345 / 384
such pairs, with 21 pairs both correct.

Example from training: red circle, yellow square, green triangle, left to right.
The model answers green to both “right of the circle?” (target yellow) and
“right of the square?” (target green).

This strongly supports a partial solution that does not reliably use the
circle/square distinction. Succeeding on two-thirds of questions and guessing
between two candidates on the other third would score 83.33%, close to the
observed training accuracy. It does **not** prove whether the bottleneck is visual
recognition, shape-word binding, or another learned computation.

The actual renderer was inspected: a size-6 circle differs from its square by
only four corner pixels; at size 8, by 12 pixels. The shapes are distinct, but
this is a plausible source of difficulty. No renderer change was made.

### Layout transfer and confidence

Training accuracy spans 80.56–84.03% across 16 geometries. Validation accuracy
across its four geometries is 45.83%, 66.49%, 44.44%, and 76.22% in sorted geometry
order. On `validation:g002`, all 144 right-target answers instead name the middle
object's color. Its bounding boxes are `(4,1,8)`, `(15,1,8)`, `(25,2,6)` in
`(left,top,size)` order; training objects never start above row 6. Another poor
layout is lower down, so vertical coverage alone does not explain the failure.
The 2,304 validation questions reuse only four layouts, not 2,304 independent
spatial configurations.

Every validation prediction is a color present in the scene; training has only
three absent-color predictions. Left/right aggregate performance is similar.
Of 962 validation errors, 523 have confidence at least 90%, compared with 27 of
1,575 training errors. Mean error confidence is 84.58% on validation versus
60.80% on training, consistent with the worsening validation loss.

These results identify incomplete training-set learning and a further layout
transfer gap. They do not establish a recurrence advantage or disadvantage.
Milestone 2 remains incomplete and the version stays unchanged. Preserve the
original baseline and diagnostic archives. The next bounded experiment is
[direct shape grounding](milestone2_shape_grounding.md), with its own protocol;
the original relational gates remain unchanged.
The direct-grounding run is now complete; its [results](milestone2_shape_grounding.md#results)
motivate a [geometry-diversity follow-up](milestone2_geometry_diversity.md).

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
