# Relational training and dependence controls

The Milestone 2 training path supports the validated three-object relational
corpus, fixed recurrence depth, epoch-boundary checkpoints, and frozen-model
image/question controls. The [first baseline protocol](milestones/milestone2.md)
is fixed at **10 epochs, R=2**, with validation acceptance gates recorded before
training. The experiment is pending; results will be recorded with that protocol.

## Importable notebooks

The repository includes two unexecuted notebooks:

| Notebook | Purpose |
| --- | --- |
| [CUDA smoke check](../notebooks/kaggle_relational_smoke.ipynb) | Small-corpus training, resume, and validation controls; run this first. |
| [Milestone 2 baseline](../notebooks/kaggle_milestone2_baseline.ipynb) | Fixed ten-epoch run or continuation, full validation controls, and the recorded acceptance gates. |

Commit and push the notebook and package changes to GitHub before importing them.
Kaggle supports notebook imports from GitHub and external URLs; see its
[import announcement](https://www.kaggle.com/product-announcements/572642).
Use these URLs in Kaggle's notebook import workflow once the files are pushed:

- [Smoke notebook on GitHub](https://github.com/Krailon/multi-modal-loop-llm/blob/milestone2/notebooks/kaggle_relational_smoke.ipynb)
- [Baseline notebook on GitHub](https://github.com/Krailon/multi-modal-loop-llm/blob/milestone2/notebooks/kaggle_milestone2_baseline.ipynb)

The links and notebook `REPO_REF` initially use `milestone2`. If running after a
branch rename/removal, use the appropriate branch or exact commit in both the
import URL and settings. The notebook import and its cloned package code should
come from the same revision. For a recorded run, retain the resolved commit.

Enable a GPU and internet access, review the first settings cell, then run cells
in order or use Run All. Setup clones the requested revision, installs the package
without its Torch extra, and checks CUDA. An existing checkout must be clean and
match the requested revision; it is never silently switched to a newer branch
tip. The notebooks call the existing scripts through
`multimodal_loop.train.kaggle`, which streams subprocess output and saves logs.
`multimodal_loop.eval.relational_protocol` validates report provenance/counts and
applies the six documented gates. No model or training logic is duplicated in
notebook cells.

Use a fresh smoke output directory each time. Start the baseline from scratch
after the smoke check passes. The baseline automatically resumes a local
`training/last.pt` for only the remaining epochs, or skips training at epoch ten.
It rejects incompatible settings and checkpoints beyond the agreed budget.
To continue from a previous session, set `RESUME_CHECKPOINT` to the restored
checkpoint and use a fresh `RUN_ROOT`. Pin `REPO_REF` to the original code commit
and preserve the original archive and provenance too.

Existing validation reports are reused only when checkpoint/manifest hashes,
settings, progress, counts, and shuffle summaries agree. The gate assessment is
saved in `validation/acceptance.json`; failure never triggers additional training.
The final cell creates an artifact ZIP outside the run directory and displays a
download link. Retain that archive (also available among notebook output files)
before the session ends. No notebook evaluates the research test split.

Notebook structure, Python cells, orchestration, and gate calculations are tested
locally with mocked script calls. Actual notebook execution on Kaggle GPU remains
the hardware check; the notebooks contain no precomputed experimental outputs.

## Kaggle baseline

Run these shell commands from the repository root with the package installed in
an environment that already provides CUDA PyTorch. Use a Bash notebook cell
(`%%bash`) or a shell; retain the existing accelerator-specific Torch installation.
Training and evaluation select one GPU, `cuda:0`. The explicit YAML is model configuration;
training settings are supplied separately on the command line.

After the separate smoke check below succeeds, generate the research corpus once
in a fresh directory. Keep this manifest for the entire run and its controls:

```bash
python scripts/generate_relational_data.py \
  --image-size 32 --object-sizes 6 8 --seed 0 \
  --train-geometry-count 16 --validation-geometry-count 4 --test-geometry-count 4 \
  --output-dir /kaggle/working/milestone2_baseline/data

mkdir -p /kaggle/working/milestone2_baseline/provenance
git rev-parse HEAD > /kaggle/working/milestone2_baseline/provenance/revision.txt
git status --short > /kaggle/working/milestone2_baseline/provenance/worktree.txt
cp configs/relational_baseline.yaml /kaggle/working/milestone2_baseline/provenance/model.yaml

python scripts/train_relational.py \
  --manifest /kaggle/working/milestone2_baseline/data/manifest.json \
  --model-config configs/relational_baseline.yaml \
  --epochs 10 --batch-size 32 --recurrence-depth 2 \
  --learning-rate 0.001 --weight-decay 0 --seed 0 --device cuda:0 \
  --output-dir /kaggle/working/milestone2_baseline/training
```

Use a committed revision containing the protocol and configuration. Retain the
console output alongside artifacts; if the tracked worktree has local changes,
record the actual diff as well so that the revision does not misidentify the code.
The script prints runtime details and the manifest hash, also saved in settings.

### Resume an interrupted baseline

Read `completed_epochs` from the checkpoint before resuming. This read does not
construct a model or restore random streams:

```bash
python - <<'PYRESUME'
import torch

checkpoint = torch.load(
    "/kaggle/working/milestone2_baseline/training/last.pt",
    map_location="cpu", weights_only=True,
)
completed = checkpoint["completed_epochs"]
print(f"Completed epochs: {completed}; remaining budget: {10 - completed}")
PYRESUME
```

For example, **only if four epochs are complete**, run the following six additional
epochs. Replace `6` with `10 - completed_epochs` for your checkpoint. If ten epochs
are already complete, proceed directly to validation controls. A checkpoint beyond
ten epochs is not the agreed baseline endpoint.

```bash
python scripts/train_relational.py \
  --resume /kaggle/working/milestone2_baseline/training/last.pt \
  --epochs 6 --device cuda:0 \
  --output-dir /kaggle/working/milestone2_baseline/training
```

Resume only at epoch boundaries on the saved backend. For a checkpoint restored
from a previous session, use its actual path for `--resume` and a writable output
directory; embedded settings, manifest, and history are authoritative. Retain the
entire run directory outside the transient session, including the source manifest
and provenance files. Never resume the research baseline from a smoke checkpoint.

### Separate CUDA smoke check

This exercises training, checkpoint resume, and controls on a smaller corpus.
It establishes successful execution, not an accuracy target or research result.
Run it once before the fresh baseline; do not tune settings using its metrics.

```bash
python scripts/generate_relational_data.py \
  --image-size 32 --object-sizes 6 8 --seed 1 --preview-count 1 \
  --train-geometry-count 1 --validation-geometry-count 1 --test-geometry-count 1 \
  --output-dir /kaggle/working/milestone2_smoke/data

python scripts/train_relational.py \
  --manifest /kaggle/working/milestone2_smoke/data/manifest.json \
  --model-config configs/relational_baseline.yaml \
  --epochs 1 --batch-size 32 --recurrence-depth 2 \
  --learning-rate 0.001 --weight-decay 0 --seed 1 --device cuda:0 \
  --output-dir /kaggle/working/milestone2_smoke/training

python scripts/train_relational.py \
  --resume /kaggle/working/milestone2_smoke/training/last.pt \
  --epochs 1 --device cuda:0 \
  --output-dir /kaggle/working/milestone2_smoke/training

python scripts/evaluate_relational.py \
  --checkpoint /kaggle/working/milestone2_smoke/training/last.pt \
  --split validation --batch-size 32 --shuffle-seeds 0 --device cuda:0 \
  --output-dir /kaggle/working/milestone2_smoke/validation
```

Confirm finite losses, completion at epoch 2 / 36 optimizer steps, and a controls
report covering 576 QA examples. Failure here calls for a hardware/runtime
investigation before the research run. The research baseline still starts from
scratch with seed 0 and the full 16/4/4 corpus.

## Training behavior and artifacts

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
selection occurs. The agreed baseline uses ten total epochs, including any
completed epochs before a resume.
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
  --checkpoint /kaggle/working/milestone2_baseline/training/last.pt \
  --split validation --batch-size 32 --shuffle-seeds 0 1 2 3 4 --device cuda:0 \
  --output-dir /kaggle/working/milestone2_baseline/validation
```

The default split is validation, batch size comes from the checkpoint, and shuffle
seeds default to `0 1 2 3 4`. Evaluation accepts `--batch-size`, `--shuffle-seeds`,
and `--device`; recurrence depth is locked to the saved training setting.
`--split test` is available for the later explicitly authorized frozen-checkpoint
evaluation. Training never evaluates the test split. No research test evaluation
is part of the first baseline run. Apply the
[recorded validation gates](milestones/milestone2.md#validation-acceptance-gates)
to the unrounded JSON metrics from the epoch-10 checkpoint. All gates must pass
before advancing to the later frozen-test step; a miss is recorded without
automatic extra training or changes to the thresholds.

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
