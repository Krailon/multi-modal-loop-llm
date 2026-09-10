# Milestone 2 — Direct-grounding training-budget follow-up

## Status and question

**Protocol specified; not yet run.** This is a separate experiment, preserving the
[completed geometry-diversity run](milestone2_geometry_diversity.md#results).
Does more training improve circle/square grounding on the broader corpus, or
mainly reinforce its partial solution? Geometry diversity raised validation
accuracy from 56.25% to 66.84%, but circle/square pair accuracy barely improved.

## Fixed protocol

| Setting | Reference | Follow-up |
| --- | ---: | ---: |
| Training geometries / images / QAs | 128 / 18,432 / 55,296 | Unchanged |
| Validation geometries / QAs | 4 / 1,728 | Unchanged |
| Updates | 2,880 | 5,760 |
| QA presentations | 92,160 | 184,320 |
| Mean presentations per QA | 1⅔ | 3⅓ |
| Full passes plus batches | 1 + 1,152 | 3 + 576 |

Use the exact derived manifest with SHA256
`047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975`.
Keep the original source manifest and derivation; validation/test records and
ordering are unchanged. At the new budget, 36,864 QAs appear three times and
18,432 appear four times. Recompute exact exposure counts by geometry, queried
shape, color and size using the existing deterministic stream.

Use fresh seed-0 weights, batch size 32, per-pass shuffling with
`random.Random(f"0:train:{epoch}")`, and the unchanged relational model config:
172,928 parameters, 32×32 RGB, patch size 8, width 64, four heads, one block each
in prelude/core/coda, R=2, dropout 0. Retain six-token direct questions, vocabulary,
renderer, float32, single CUDA device, AdamW lr 0.001, weight decay 0,
betas (0.9, 0.999), eps 1e-8, `foreach=False`, `fused=False`.

Validate at steps 0, 288, …, 5,760. Evaluate the final checkpoint irrespective of
history. No resume, learning-rate change, early stopping, earlier-checkpoint
selection, automatic extension, recurrence sweep, or test inference. This fresh
run repeats the original training prefix; it does not continue reference weights.
Training updates and presentations double; report actual runtime separately from
training counts and full-corpus diagnostic costs. Record hardware/software
changes; identical CUDA trajectories across environments are not guaranteed.

## Evaluation and interpretation

Reuse frozen diagnostics on all training and validation QAs: per-shape, geometry,
position, size, all-three, circle/square pair, identical-prediction and position
confusion metrics. Preserve original/added training subsets, every prediction,
and the error preview. Scene metadata remains inspection/supervision only.

Keep blank-image, shuffled-image and shuffled-question validation controls with
seeds 0–4 and original recipient scoring targets. Retain the nine criteria:
≥95% training accuracy for each shape, ≥90% validation accuracy for each shape,
and ≥30-percentage-point overall gaps against all three controls. These are
unchanged direct-grounding criteria, not relational milestone completion gates.

Compare against the final **2,880-update geometry-diversity checkpoint**, SHA256
`49012c1c488478c7eb66e27b979b59c22a47a92d696120ad9279718afbd71456`.
Both runs use identical training and validation corpora. `comparison.json` reports
current-minus-reference accuracy, loss, all-three and circle/square pair accuracy,
identical-prediction frequency, per-shape accuracy, original/added training
subsets, and validation control gaps. Lower loss and identical-prediction
frequency are favorable but cannot establish grounding alone.

- Improved circle/square training and validation performance supports a benefit
  from additional training at this budget.
- Better training fit with stalled validation points to a remaining transfer gap.
- Poor circle/square training and pair accuracy means this budget increase did
  not resolve learning; it does not prove an architectural limitation.
- Overall improvement mainly through triangles does not resolve circle/square
  grounding. Inspect the full trajectory without replacing the final result.

This is one seed and four validation geometries. It neither establishes broad
layout generalization nor tests a recurrence advantage. Future evaluation breadth
or interventions require separate specifications; no further extension is implied.

## Kaggle workflow and artifacts

Import [the notebook](../../notebooks/kaggle_milestone2_training_budget.ipynb)
from GitHub after committing/pushing. Enable GPU and internet, select a committed
`REPO_REF` (commit SHA preferred), and attach the completed
`milestone2_geometry_diversity_artifacts.zip`. Set `REFERENCE_SOURCE` to that ZIP
or its extracted directory; use fresh checkout/output paths and run all cells.
The notebook preserves Kaggle's installed PyTorch and uses `cuda:0`.

The helper API is `prepare_training_budget(repo, root, source)`,
`run_training_budget(run)`, and `archive_training_budget(run)` in
`multimodal_loop.eval.kaggle_training_budget`. Reference artifacts are audited
before training, and reference weights never initialize the model. Existing
training/evaluation commands and completed-experiment defaults are unchanged.

The default output archive is `milestone2_training_budget_artifacts.zip`:

- `data/`: exact manifests, derivation, coverage and new planned exposures.
- `training/`: fresh checkpoint, settings, history and actual-budget exposures.
- `diagnosis/`: full predictions, summary, controls, comparison and HTML preview.
- `provenance/`: fixed protocol, reference reports, revision/runtime and hashes.
- `logs/`: training and evaluation output.

Staged reference weights are excluded. Bring the archive back for review; no
results or completion claims are recorded until that run has been evaluated.
