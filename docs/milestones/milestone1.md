# Milestone 1 — Synthetic vision: results and close-out

## Verdict

**Complete, scoped to single-object color grounding on held-out layouts.**

The milestone's stated success criterion is: “The model learns meaningful visual
grounding without a pretrained vision encoder.” A model initialized from scratch
achieves 100% correct-image accuracy on validation and on the frozen test split,
while shuffled and blank images reduce accuracy to approximately 25%. This meets
the criterion for this deliberately small color-perception task.

The close-out review found **no blocking correctness or evidence-integrity defects**.
Milestone 2 preparation is next: multi-object scenes and relational questions,
followed by experiments testing whether additional recurrence helps.

## Review evidence

Reviewed implementation revision: `8d1e3b37719fef10f29d69eac21dc9bc5586f78c`.
This identifies the code reviewed, not an independently recorded original training
revision. This close-out changes documentation only; it performs no new training
or model evaluation on the research test set.

| Area | Finding and evidence |
| --- | --- |
| Rendering and labels | Deterministic hard-edged RGB geometry and exact color labels are independently checked in [rendering tests](../../tests/test_synthetic_shapes.py). |
| Split integrity | The saved manifest passes [manifest validation](../../src/multimodal_loop/data/multimodal.py); 256 training, 64 validation, and 64 test layouts are disjoint. Each selected layout occurs once in each of four colors. Rendering tests also cover pixel disjointness. |
| Metadata exclusion and supervision | The [collator](../../src/multimodal_loop/data/collator.py) consumes pixels, question, and explicit answer only. [Batching tests](../../tests/test_color_batching.py) check metadata exclusion and final-question-logit target alignment. |
| Recurrent leakage prevention | The [model](../../src/multimodal_loop/model/model.py) applies the same attention mask at every stage and repetition. [Model tests](../../tests/test_model.py) check protected states after every recurrence; batching tests compare answer-supervised and question-only predictions. |
| Training protocol | The [dataset trainer](../../src/multimodal_loop/train/synthetic.py) visits every example once per epoch at explicit fixed depth. [Training tests](../../tests/test_synthetic_training.py) verify shuffle reproducibility, partial batches, weighted metrics, and optimizer step counts. The CLI uses a fixed epoch budget and no best-checkpoint selection. |
| Evaluation and controls | The [evaluator](../../src/multimodal_loop/eval/synthetic.py) forwards only question IDs and images; argmax spans the vocabulary. [Control tests](../../tests/test_image_controls.py) verify that only pixels change, original targets remain, donors stay within the chosen split, and state is preserved. |
| Artifact consistency | Checkpoint and manifest hashes match both control reports. Embedded corpus, model/training settings, progress, metric history, all donor permutations, pairing fractions, and aggregate metrics agree with the saved files. Final validation metrics exactly equal the checkpoint's epoch-10 validation record. |
| Deterministic resume | Training tests establish exact CPU model, optimizer, loss/history, and RNG equality across epoch-boundary resume with dropout, partial batches, and fresh processes. This is infrastructure verification, not a second independently trained research baseline. |

Close-out checks: **675 tests passed, 4 accelerator tests skipped**; `ruff check .`,
formatting checks, and `git diff --check` passed. The skipped tests require CUDA or
XLA hardware. No new dependency, model component, or training behavior was introduced.

## Experiment specification

The task is exactly **“What color is the object?”**, with red, green, blue, or yellow
as a one-token answer. Each 32×32 image contains one square, circle, or upright
triangle on black. Bounding-box sizes are 8, 12, or 16 pixels with a one-pixel canvas
margin; pixels are saturated RGB values without antialiasing. Corpus seed is 0.

| Split | Examples | Color-independent layouts | Examples per color |
| --- | ---: | ---: | ---: |
| Training | 1,024 | 256 | 256 |
| Validation | 256 | 64 | 64 |
| Test | 256 | 64 | 64 |

A layout is `(shape, size, left, top)`. Layouts are allocated without replacement,
with all four colors represented per layout. Shape/color categories recur across
splits; the held-out distinction is layout combinations, not unseen categories.
Manifest format is version 1. Stored record order is authoritative.

The model uses direct patch embeddings, learned text/position/modality embeddings,
one transformer prelude, a weight-shared recurrent core, one coda, final LayerNorm,
and an untied language-model head. There is no pretrained vision encoder.
The resolved model configuration is:

```json
{
  "vocab_size": 10,
  "max_seq_len": 128,
  "d_model": 64,
  "n_heads": 4,
  "d_ff": 256,
  "n_prelude_layers": 1,
  "n_recurrent_layers": 1,
  "n_coda_layers": 1,
  "recurrence_depth": 2,
  "image_size": 32,
  "patch_size": 8,
  "num_channels": 3,
  "dropout": 0.0,
  "layer_norm_eps": 1e-05
}
```

Tokenizer version 1 has IDs 0–5 for `What color is the object ?` and IDs 6–9
for `red green blue yellow`, with no special tokens. Six question tokens follow
16 image patches. Training includes the answer input behind a prefix attention
barrier; only the final question logit predicts a supervised answer. Evaluation
omits the answer input entirely while retaining the image/question prefix.

Training seed is **0**, batch size **32**, recurrence depth **R=2**, and the budget
is **10 epochs / 320 AdamW updates / 10,240 example presentations**. AdamW uses
learning rate **0.001**, weight decay **0**, default betas `(0.9, 0.999)` and epsilon
`1e-8`, with `foreach=False` and `fused=False`. No scheduler, early stopping, or
validation-based checkpoint selection is used. Epoch `e` shuffles with local
`random.Random(f"0:train:{e}")`; the final checkpoint is the epoch-10 `last.pt`.

The baseline used Python **3.12.3**, PyTorch **2.14.0+cpu**, float32, and one CPU
compute thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). Validation occurred before
training and after every epoch. Losses below are example-weighted cross-entropy;
training losses average pre-update predictions throughout each epoch.

## Training and validation history

| Epoch | Updates | Training loss | Validation loss | Correct/256 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | — | 2.540769 | 64 | 25.0000% |
| 1 | 32 | 1.559847 | 1.351554 | 136 | 53.1250% |
| 2 | 64 | 0.739264 | 0.127205 | 255 | 99.6094% |
| 3 | 96 | 0.057924 | 0.026079 | 256 | 100.0000% |
| 4 | 128 | 0.019351 | 0.015843 | 256 | 100.0000% |
| 5 | 160 | 0.013093 | 0.011464 | 256 | 100.0000% |
| 6 | 192 | 0.009754 | 0.008768 | 256 | 100.0000% |
| 7 | 224 | 0.007613 | 0.006949 | 256 | 100.0000% |
| 8 | 256 | 0.006120 | 0.005650 | 256 | 100.0000% |
| 9 | 288 | 0.005033 | 0.004689 | 256 | 100.0000% |
| 10 | 320 | 0.004216 | 0.003961 | 256 | 100.0000% |

Accuracy reached 100% at epoch 3 and remained there through epoch 10. The full
predetermined budget was completed without tuning or selecting an earlier epoch.

## Image-control protocol and results

The same frozen epoch-10 checkpoint is used for every condition at R=2 and batch
size 32. Only image pixels change: questions, original answer targets, recipient
order, masks, image-token count, and positional/modality layout are unchanged.
Blank images are zero tensors, not omitted image tokens. Shuffling permutes the
entire selected split with local `random.Random(seed).shuffle` for seeds **0–4**,
using each donor once. Self-pairings and same-color coincidences are retained;
labels never influence donor selection. Donors never cross split boundaries.

Correct and blank images are each evaluated once per split; all five predetermined
shuffles are reported. Argmax covers the full vocabulary and non-color predictions
count as errors. All conditions below produced **zero non-color predictions**.

The frozen test protocol was established after validation controls, then applied
to all 256 previously untouched test examples without further tuning. The test
split has now been evaluated and must no longer be described as untouched.

### Validation controls

| Images | Correct/total | Accuracy | Cross-entropy | Same-color pairing fraction |
| --- | ---: | ---: | ---: | ---: |
| Correct | 256/256 | 100.0000% | 0.003961 | — |
| Shuffled, seed 0 | 63/256 | 24.6094% | 5.599248 | 24.6094% |
| Shuffled, seed 1 | 56/256 | 21.8750% | 5.829057 | 21.8750% |
| Shuffled, seed 2 | 77/256 | 30.0781% | 5.167806 | 30.0781% |
| Shuffled, seed 3 | 55/256 | 21.4844% | 5.828571 | 21.4844% |
| Shuffled, seed 4 | 70/256 | 27.3438% | 5.385539 | 27.3438% |
| Blank | 64/256 | 25.0000% | 4.397621 | — |

Shuffled mean accuracy: **25.0781%** (range 21.4844%–30.0781%).
Mean cross-entropy: **5.562044** (range 5.167806–5.829057).
Correct-image accuracy exceeds the shuffled mean by **74.9219
percentage points**, and blank accuracy by **75 percentage points**.

### Test controls

| Images | Correct/total | Accuracy | Cross-entropy | Same-color pairing fraction |
| --- | ---: | ---: | ---: | ---: |
| Correct | 256/256 | 100.0000% | 0.003840 | — |
| Shuffled, seed 0 | 73/256 | 28.5156% | 5.348914 | 28.5156% |
| Shuffled, seed 1 | 61/256 | 23.8281% | 5.656441 | 23.8281% |
| Shuffled, seed 2 | 60/256 | 23.4375% | 5.737998 | 23.4375% |
| Shuffled, seed 3 | 61/256 | 23.8281% | 5.702248 | 23.8281% |
| Shuffled, seed 4 | 60/256 | 23.4375% | 5.718033 | 23.4375% |
| Blank | 64/256 | 25.0000% | 4.397621 | — |

Shuffled mean accuracy: **24.6094%** (range 23.4375%–28.5156%).
Mean cross-entropy: **5.632727** (range 5.348914–5.737998).
Correct-image accuracy exceeds the shuffled mean by **75.3906
percentage points**, and blank accuracy by **75 percentage points**.

Every shuffled accuracy equals its realized same-color pairing fraction. This is
consistent with predictions following the donor image's color. The five shuffles
measure pairing variation on one split, not five independent training runs or
independent datasets. Blank and shuffled performance are near the balanced-color
baseline of 25%, while correct images achieve 100% on both held-out splits.

## Provenance and artifact availability

- Manifest SHA256: `4b15b07dd6728ae3a3f37ca4fe900288e184ed2ae6ba2261ea515ffb357261a2`.
- Frozen checkpoint SHA256: `08fa29639f7566829c0c4e5172b7945b95462c81c4f6fd483fcf55ad6c0b0047`.

| Local artifact | Contents |
| --- | --- |
| `outputs/synthetic_shapes/manifest.json` | Original ordered corpus; matches the training copy and embedded corpus. |
| `outputs/color_baseline/settings.json` | Resolved model, training, tokenizer, runtime, and thread settings. |
| `outputs/color_baseline/manifest.json` | Exact training corpus file. |
| `outputs/color_baseline/metrics.json` | All epoch metrics; agrees with checkpoint history. |
| `outputs/color_baseline/last.pt` | Frozen epoch-10 model, optimizer, progress, corpus, and RNG state. |
| `outputs/color_controls/controls.json` | Validation results, permutations, pairing fractions, and provenance. |
| `outputs/color_test_controls/controls.json` | Frozen test results and corresponding provenance. |

The prior frozen-test verification established that all training artifacts and the
validation report were byte-for-byte unchanged by test evaluation. This close-out
rechecked the shared checkpoint/manifest hashes and report consistency.
Raw outputs remain **ignored local files**, not bundled with this committed report.
The Markdown preserves the results and provenance; a clean clone will need the
original artifacts supplied separately or a new reproduction run. Checkpoint-file
hashes identify the original artifact and are not promised for newly serialized runs.

## Reproduction commands

Run from the repository root in the installed environment described above.
Use fresh directories to keep the original evidence frozen. The following commands
are instructions, not additional experiments executed during close-out:

```bash
python scripts/generate_synthetic_data.py --output-dir outputs/milestone1_reproduction/data

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_synthetic.py \
  --manifest outputs/milestone1_reproduction/data/manifest.json \
  --output-dir outputs/milestone1_reproduction/training \
  --epochs 10 --batch-size 32 --recurrence-depth 2 \
  --learning-rate 0.001 --weight-decay 0 --seed 0 --device cpu

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/evaluate.py \
  --checkpoint outputs/milestone1_reproduction/training/last.pt \
  --split validation --batch-size 32 --shuffle-seeds 0 1 2 3 4 \
  --device cpu --output-dir outputs/milestone1_reproduction/validation

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/evaluate.py \
  --checkpoint outputs/milestone1_reproduction/training/last.pt \
  --split test --batch-size 32 --shuffle-seeds 0 1 2 3 4 \
  --device cpu --output-dir outputs/milestone1_reproduction/test
```

Use the reviewed code revision and recorded model defaults for reproduction.
Exact numerical replay depends on matching hardware/software and thread settings.
The generator records software versions, so regenerating under another environment
can change manifest bytes/hash even when scene descriptions agree.

## Nonblocking limitations and handoff

- One training seed and one corpus were studied. Five image shuffles do not measure
  variation across model initialization or independent training runs.
- All questions are identical. The result demonstrates image-dependent color
  perception, not understanding different questions or conditioning on their meaning.
- Saturated colors, black backgrounds, and single objects permit simple global color
  statistics to solve the task. The result does not establish localization, shape
  recognition, relationships, multi-hop reasoning, or real-image generalization.
- No dense baseline or recurrence comparison was run. R=2 working does not imply
  that recurrence improves accuracy or computational efficiency.
- This research path is CPU-validated. The earlier Kaggle sign-off covers Milestone 0
  infrastructure; Milestone 1 accelerator experiments and TPU hardware validation
  remain outside this result's evidence.
- Raw artifacts are local-only, and the full research run was not independently
  repeated during review. Those are reproducibility limits, not hidden replications.

These limitations constrain the claim rather than block the stated small baseline.
Preserve the corpus and checkpoint as the Milestone 1 reference. Begin Milestone 2
with deterministic multi-object scenes and relational labels, then evaluate whether
additional recurrent steps help tasks that require additional reasoning.

[Return to the research overview](../../README.md)
