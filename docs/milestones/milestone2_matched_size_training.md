# Milestone 2 — Matched size-variant training

## Status and experimental question

**Protocol specified; research training has not yet run.** The
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
weights are excluded. Bring the archive back for review; no research result is
recorded yet.
