# Milestone 2 — Bounded quartet training-fit check

## Status and question

**Protocol specified; research training has not yet run.** The
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
Reference weights are excluded. Bring the archive back for review; this document
records a protocol, not a completed research result.
