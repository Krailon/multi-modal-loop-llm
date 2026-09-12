# Milestone 2 — Bounded quartet training-fit check

## Status and question

**Completed; all four training-fit criteria passed.** The unchanged model achieved
100% accuracy and correct answers across all four sizes for every family; see
[results](#results). The
[matched-size training result](milestone2_matched_size_training.md#results) did not
resolve size-dependent circle/square confusion at its fixed budget. Can the
unchanged model reliably fit a small, balanced collection of valid matched size
quartets when it sees every example repeatedly?

This diagnostic measures training fit only. A pass does not establish held-out
shape grounding, relational reasoning, or a recurrence advantage. A failure
applies to this protocol and budget; it does not establish architectural
impossibility. Milestone 2 remains incomplete.

## Deterministic training population

Read only the corpus from `milestone2_matched_size_training_artifacts.zip`, with
manifest SHA256
`25071dc462e1e8bdadee82ca7b0531cdde871c4f9735613532f41210c9192828`.
Verify that identity and its deterministic derivation before selection. No
reference checkpoint weights or prediction results participate in selection.

Group eligible complete families by ordered object origins and triangle size.
A complete group contains all six shape orders and all 24 assignments of three
distinct colors from four available colors. Select the lexicographically first
complete group, without examining performance: origins **(1,4), (10,4), (20,4)**,
with **triangle size 8**. This is one fixed arrangement of object origins; the
bounding-box sizes vary within it.

Sort families by the ordered `(shape, color)` identities, then use the fixed
condition order: circle/square sizes **6/6, 6/8, 8/6, 8/8**. Retain all four members
of every family and record their indices in the source matched-size corpus.
Each image has all three direct shape-color questions in the existing tokenizer
order (square, circle, triangle).

| Population measure | Count |
| --- | ---: |
| Families | 144 |
| Unique images | 576 |
| Direct questions | 1,728 |
| Images per size condition | 144 |
| Images per shape order | 96 |
| Images per spatial color assignment | 24 |
| Questions per queried shape | 576 |
| Questions per answer color | 432 |

Use existing pixels and shape masks, 32×32 RGB, fixed origins and colors, blank
canvas margins and at least one blank column between object boxes. Triangle size
stays 8. As in matched-size training, vertical centers need not remain aligned.
There is no edge-contact relaxation. Verify training origins do not match any
reserved validation/test origins. The manifest embeds source provenance, but
only the selected training records are exposed to the experiment's dataset.
No validation or test inference is authorized.

The local preflight subset manifest SHA256 is
`86bdb774cef97e6b76752f016edd2b73473e8c989b77a7e2cedcb4e295894180`.
Selection and exact 40-visit exposure counts were verified locally without
training the research model.

## Fixed training procedure

Initialize fresh seed-0 weights on CPU, then move to the selected CPU/CUDA device.
Use unchanged `configs/relational_baseline.yaml`: 172,928 parameters, direct
8×8 image patches, width 64, four heads, one prelude/core/coda block each, R=2,
dropout 0, float32. Keep batch size 32, AdamW lr 0.001, weight decay 0,
betas (0.9,0.999), eps 1e-8, `foreach=False`, `fused=False`, and answer-only loss.
Use the existing six-token questions and deterministic shuffle
`random.Random(f"0:train:{epoch}")` for each pass.

Train for exactly **2,160 updates / 69,120 QA presentations**. A full pass is
54 batches, so every QA receives exactly **40 presentations**. Monitor the entire
training-fit population at step zero and every 54 updates, for 41 records.
Persist these metrics under `training_fit`; interval optimization loss remains
under `train`. Neither population is labeled validation.

No resume, early stopping, budget extension, learning-rate change, family batching,
consistency loss, or best-checkpoint selection. Evaluate the final frozen
checkpoint on the same 1,728 training questions, independent of its trajectory.
The monitoring and final diagnostic passes are evaluation, not gradient updates.

## Final assessment and interpretation

Both requirements must hold at the final checkpoint:

- **At least 99% accuracy for each queried shape:** at least 571/576 correct for
  square, circle and triangle separately.
- **At least 95% correctly invariant families:** at least 137/144 families retain
  both circle/square answers correctly across all four size conditions. Each
  successful family therefore requires eight correct circle/square answers.

Values are assessed without rounding. Stable wrong predictions do not count as
correct invariance. These four pass/fail checks are separate from all earlier
protocols and Milestone 2 completion gates.

Record overall/per-shape accuracy and loss, full training-fit history, size-condition
circle/square pair accuracy, relative-size swaps and other-color selections,
per-family correctness, and a preview of confident errors. Preserve raw
predictions so grouping and labels can be checked independently.

Success would establish that this model and training procedure can fit the
balanced distinction on repeatedly seen examples. It would not distinguish
memorization from a transferable shape rule. Failure would motivate investigation
of optimization, representation or implementation without uniquely identifying
which caused it. This single-seed diagnostic deliberately removes most layout
variation; its budget is not a matched comparison with earlier research runs.
Decide any follow-up separately after reviewing the recorded result.

## Interfaces and Kaggle workflow

Commit/push the implementation, then import
[the Kaggle notebook](../../notebooks/kaggle_milestone2_quartet_fit.ipynb). Enable
GPU and internet, set a committed `REPO_REF` (SHA preferred), and attach the
matched-size archive as `REFERENCE_SOURCE`. ZIPs or extracted run directories are
accepted. Use fresh checkout/output paths; retain Kaggle's installed PyTorch.
The research run uses one CUDA device and float32; runtime and CPU thread count
are recorded with the artifacts.

`prepare_quartet_fit(repo, root, source)` verifies the source and writes the subset
manifest, derivation, exposure counts and protocol. `run_quartet_fit(run)` invokes
fresh training and final frozen diagnostics. `archive_quartet_fit(run)` verifies
recorded hashes and packages the completed run. These helpers live in
`multimodal_loop.eval.kaggle_quartet_fit`.

`scripts/train_quartet_fit.py` accepts `--manifest`, `--model-config`, `--output-dir`,
`--device`, `--max-steps` and `--evaluation-interval`. Research execution enforces
the fixed settings; `--smoke` explicitly permits test configurations/budgets and
marks their artifacts. `scripts/evaluate_quartet_fit.py` accepts `--checkpoint`,
`--output-dir` and `--device`. Neither script accepts a held-out split.

Manifest and checkpoint kind is `quartet_training_fit`, version 1. Checkpoints
preserve model/optimizer state, configuration, tokenizer, corpus identity,
history/progress and randomness state. Frozen loading does not restore training
randomness or provide resume. Existing checkpoint formats remain unchanged.

The archive `milestone2_quartet_fit_artifacts.zip` contains data/provenance,
`training/last.pt`, settings, history, exposure counts, final training-fit
predictions and summary, family diagnostics, inspection HTML, hashes and logs.
Reference weights are excluded. The completed archive was reviewed as recorded below.


## Results

The completed `milestone2_quartet_fit_artifacts.zip` passes all four specified
training-fit criteria. Fresh seed-0 weights reached perfect accuracy on the
balanced training population with the unchanged model, R=2 and optimizer.
Training completed the fixed 2,160 updates / 69,120 QA presentations: 40 complete
passes over 576 images and 1,728 questions.

### Provenance and audit

The archive reports clean revision
`95c1d24e6b1f9f95b4b809ca25794527448f4705`, PyTorch 2.10.0+cu128, CUDA 12.8,
one Tesla T4, float32 and two CPU threads.

Local review verified all 14 hashes in `provenance/final.json`, regenerated the
specified subset, checked its derivation and exact 40-visit exposure counts, and
validated checkpoint configuration, history and training progress against the
fixed protocol. All 1,728 saved predictions were checked against scene labels and
summary metrics. Family correctness and acceptance criteria were independently
recomputed from those records. No new model inference was performed during review;
the experiment evaluated training fit only, with no validation or test inference.

- Manifest SHA256: `86bdb774cef97e6b76752f016edd2b73473e8c989b77a7e2cedcb4e295894180`
- Final checkpoint SHA256: `3e18600b01fcf94514d7a2aed9a1e4bf4d5b21958bd208966e57dee056f1fee4`
- Summary SHA256: `842b57c8edd9606db37fe49d8a8f02b1e6d487d914ffad3566de332e40d5fd9d`
- Predictions SHA256: `e9f4898240012e0dd73241ca2b95463a701ef9ac0ead71a240ff709b5e86a41a`

### Final checkpoint

| Measure | Result | Required | Outcome |
| --- | ---: | ---: | --- |
| Square accuracy | 576/576 — 100% | ≥99% | Pass |
| Circle accuracy | 576/576 — 100% | ≥99% | Pass |
| Triangle accuracy | 576/576 — 100% | ≥99% | Pass |
| Families with both circle/square answers correct across all four sizes | 144/144 — 100% | ≥95% | Pass |

Overall accuracy is **1,728/1,728 (100%)**, with final loss **0.000406125**.
All three answers are correct on every image (576/576), and there are no invalid
predictions. The four criteria are assessed on the final checkpoint, without
rounding or substituting an earlier checkpoint.

| Circle size / square size | Overall accuracy | Both circle/square answers correct |
| --- | ---: | ---: |
| 6 / 6 | 432/432 — 100% | 144/144 — 100% |
| 6 / 8 | 432/432 — 100% | 144/144 — 100% |
| 8 / 6 | 432/432 — 100% | 144/144 — 100% |
| 8 / 8 | 432/432 — 100% | 144/144 — 100% |

This includes the smaller-square/larger-circle combination that remained
particularly difficult in previous experiments. Every family satisfies correct
invariance across the four sizes; stable wrong predictions do not contribute.

### Learning trajectory

| Update | Completed passes | Training-fit accuracy |
| --- | ---: | ---: |
| 324 | 6 | 54.69% |
| 540 | 10 | 82.35% |
| 648 | 12 | 92.94% |
| 702 | 13 | 99.31% |
| 864 | 16 | 100% |
| 2,160 | 40 | 100% |

The first recorded perfect evaluation is at **update 864 / pass 16**. Accuracy
remains 100% at every subsequent recorded evaluation through pass 40. Loss falls
from 0.004539 at update 864 to 0.000406125 at the final checkpoint. Monitoring was
performed once per pass, so this identifies the first recorded perfect score,
not the exact update at which every answer became correct.

### Interpretation and next decision

The unchanged direct-patch model and optimizer can fit this balanced shape/size
distinction on repeatedly seen examples. The successful result narrows the
problem: the earlier failures do not demonstrate that the architecture cannot
represent correct answers for these combinations or that the training procedure
cannot learn them at all.

Transfer across arrangements remains unresolved. Perfect training fit can rely on
location-specific patterns or memorization; it does not establish a general
shape rule. This experiment changes layout diversity and repetition relative to
the larger-corpus experiments and does not isolate the cause of their failures.
No recurrence advantage was tested. Milestone 2 remains incomplete.

The proposed next decision is a frozen-model transfer check on complete quartets
from other training-source arrangements that this model never saw. Those scenes
would be unseen by this checkpoint, but remain part of the broader project's
training source; they should not be described as a new untouched test split.
That follow-up is not specified or run by this results record. Preserve all
protocols and artifacts, and reserve test inference for a later authorized step.
