# Relational training and dependence controls

The Milestone 2 training path supports the validated three-object relational
corpus, fixed recurrence depth, epoch-boundary checkpoints, and frozen-model
image/question controls. This infrastructure has correctness tests; a research
baseline run and numerical acceptance thresholds remain the next step. Future
experiment results belong in `docs/milestones/milestone2.md`.

## Train and resume

Generate the corpus with `scripts/generate_relational_data.py` first if needed.
The following commands illustrate usage; the epoch count is a CLI example,
not a selected research budget:

```bash
python scripts/train_relational.py \
  --manifest outputs/relational_shapes/manifest.json \
  --epochs 1 --recurrence-depth 2 \
  --output-dir outputs/relational_training

python scripts/train_relational.py \
  --resume outputs/relational_training/last.pt \
  --epochs 1 --output-dir outputs/relational_training
```

Fresh runs initialize the model from scratch. Defaults are vocabulary size 17,
image size from the manifest, and the remaining `ModelConfig` defaults: width 64,
4 heads, feed-forward width 256, one block each in the prelude, shared core, and
coda, patch size 8, and dropout 0. Training defaults are batch size 32, learning
rate 0.001, weight decay 0, seed 0, and recurrence depth 2. `--model-config` accepts
a YAML mapping of model overrides. The model must have vocabulary size exactly
17, three image channels, matching image size, and capacity for P+12 positions.

Training overrides are `--batch-size`, `--learning-rate`, `--weight-decay`,
`--seed`, and `--recurrence-depth`. An explicit depth flag takes precedence over
the model configuration's default depth. Both training and validation use that
saved training depth. `--epochs` defaults to 10 and always means additional
complete epochs in this invocation; no early stopping or best-checkpoint
selection occurs. Agree on the research budget before running the baseline.
At the default corpus size, one epoch visits **9,216 QA examples** and performs
**288 optimizer steps** with batch size 32.

`relational_loader` uses a local epoch-specific permutation of QA indices, a
private Torch loader generator, no workers, and no dropped final batch. Training
reuses `SyntheticTrainingConfig`, `train_synthetic_epoch`, and answer-only shifted
cross-entropy. Scene/query metadata never enters the model. Validation preserves
stored order and forwards only the 11 question tokens with an all-prefix mask;
the final question logit predicts the answer over the full vocabulary. Only IDs
6–9 are valid color answers. Validation records example-weighted loss, accuracy,
counts, and invalid predictions at epoch zero and after every epoch.

`--device` accepts `cpu`, `cuda`, `cuda:N`, or `xla` through the existing runtime.
Execution is float32, single process, and single device. The relational path has
CPU correctness coverage; accelerator hardware validation for this path remains
separate. The prior infrastructure CUDA sign-off is not a relational training result.

Each run writes:

- `settings.json`: model/training settings, tokenizer, manifest hash, and runtime.
- `manifest.json`: exact embedded corpus content.
- `metrics.json`: epoch-zero validation and each completed epoch's train/validation metrics.
- `last.pt`: authoritative, atomically replaced epoch-boundary checkpoint.

Relational checkpoints use kind `relational_color`, format version 1. They contain
configuration, model/AdamW state, tokenizer version and full vocabulary, exact
manifest and hash, progress, metric history, and host/selected-device RNG states.
They are distinct from Milestone 1 checkpoints. Load validates the task vocabulary,
manifest, optimizer settings, and QA-based epoch counts before restoring RNGs.

Resume uses the embedded manifest, settings, and history, even if external files
are missing or stale. Fresh-run overrides are rejected with `--resume`. A fresh
run refuses to overwrite existing run artifacts. Same-backend resume restrictions
apply; exact CPU continuation with dropout is tested across process restarts.

## Evaluate a frozen checkpoint

```bash
python scripts/evaluate_relational.py \
  --checkpoint outputs/relational_training/last.pt \
  --split validation --output-dir outputs/relational_controls
```

The default split is validation, batch size comes from the checkpoint, and shuffle
seeds default to `0 1 2 3 4`. Evaluation accepts `--batch-size`, `--shuffle-seeds`,
and `--device`; recurrence depth is locked to the saved training setting.
`--split test` is available for the later explicitly authorized frozen-checkpoint
evaluation. Training never evaluates the test split. No research test evaluation
is part of this infrastructure increment.

The evaluator writes `controls.json` and refuses to overwrite an existing report.
It records checkpoint/manifest hashes, progress, model/training/evaluation settings,
runtime, per-condition metrics, shuffle permutations, summaries, and accuracy gaps.
There are no optimizer updates; evaluation preserves weights, gradients, module
modes, and random streams.

Every condition reports overall accuracy/loss/invalid predictions plus:

- `by_question`: counts and accuracy for each of the six original question texts.
- `all_four`: fraction of image records with all four answers correct.
- `different_answer_pairs`: fraction of unordered, different-reference-answer
  question pairs for which both predictions are correct. There are five such
  pairs per image; the two questions targeting the middle object share an answer
  and their pair is excluded. Stored answers determine pairing, not slot order.

The conditions are:

| Report key | Intervention |
| --- | --- |
| `correct` | Original images, questions, and targets. |
| `blank` | Zero pixels, with original questions, targets, and layout. |
| `shuffled_images` | Whole-image record permutation within the selected split; all four recipient questions receive the same donor image. |
| `shuffled_questions` | Independent four-question permutation within each image; pixels and original answer targets stay fixed. |

Permutations are ordinary seeded shuffles, chosen without inspecting labels.
Self-pairings and answer coincidences remain. Each seed reports accuracy/loss and
its gap from correct inputs; summaries give mean/min/max accuracy and loss.
Image controls do not change question tokens or sequence layout. Question controls
replace only the 11 question IDs. Grouping remains correct across batch boundaries.
All grouped/per-question scores refer to **original recipient supervision**, even
when an intervention makes that supervision inconsistent with the supplied inputs.

Question shuffling deliberately breaks question–answer alignment. Its
`same_answer_pairing_fraction` reports how often the replacement question still
has the original answer. With answer multiplicities 2,1,1, an ideal model's expected
accuracy under ordinary question shuffling is **37.5%**, not 25%; the exact observed
fraction is reported per seed. Shuffled images can also make a recipient question
lack a valid neighbor in the donor scene. These are dependence diagnostics, not
new correctly labelled reasoning datasets.

Overall accuracy alone is insufficient: always predicting the middle object's
color can achieve **50%** while ignoring the question. That predictor scores zero
on both all-four and different-answer-pair accuracy. Interpret the correct-input
grouped metrics together with the image and question controls. These checks
establish capability evidence; they do not establish a recurrence advantage.
