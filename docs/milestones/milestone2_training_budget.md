# Milestone 2 — Direct-grounding training-budget follow-up

## Status and question

**Experiment completed and audited; five of nine criteria passed.** This is a separate experiment, preserving the
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

Staged reference weights are excluded. The completed archive was reviewed locally;
results are recorded below.


## Results

The completed `milestone2_training_budget_artifacts.zip` was reviewed against the
fixed protocol. Training used fresh seed-0 weights for 5,760 updates and 184,320
QA presentations on the unchanged 128-layout corpus. The run used PyTorch
2.10.0+cu128, CUDA 12.8, one Tesla T4, float32, and two CPU threads.

The archive reports clean revision
`91748ecc0c03f9b00eda7a0aa0b61e7b8d71e375`.

### Artifact audit

Local checks verified artifact hashes, checkpoint progress, settings, exact corpus
identity, deterministic exposure counts, saved prediction summaries, comparison
arithmetic and acceptance criteria. The recorded training/validation history
through step 2,880 exactly matches the reference history; this does not establish
that intermediate model weights were compared. No new model or test inference
was performed during review.

- Manifest SHA256: `047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975`
- Final checkpoint SHA256: `2069297aabdff24489bf2be5ef2f0c5a53989f4daaea312cd55e51a72282f8bc`
- Summary SHA256: `ab5295daf08e6844b2b011699f75275f968d5e811ba044d1d3b6100f5b4f8a23`
- Controls SHA256: `f1a2d75fc2a986bb6825766e096d4e91f6b475370d041a0193f58b546141cb8c`
- Comparison SHA256: `b7e5aad04c6f4fdfbc99a618f684e9222b4c57a36bf31a6259882d60014af6b3`

### Final checkpoint

| Metric | Training | Validation |
| --- | ---: | ---: |
| Overall accuracy | 77.51% | 74.71% |
| Loss | 0.4351 | 0.5598 |
| All-three accuracy | 57.92% | 56.08% |
| Circle/square pair accuracy | 59.28% | 56.77% |
| Identical circle/square predictions | 15.57% | 15.97% |

| Criterion | Result | Required | Outcome |
| --- | ---: | ---: | --- |
| `train_circle` | 67.18% | ≥95% | Fail |
| `train_square` | 68.09% | ≥95% | Fail |
| `train_triangle` | 97.25% | ≥95% | Pass |
| `validation_circle` | 65.28% | ≥90% | Fail |
| `validation_square` | 66.32% | ≥90% | Fail |
| `validation_triangle` | 92.53% | ≥90% | Pass |
| `correct_minus_blank` | 49.71 pp | ≥30 pp | Pass |
| `correct_minus_shuffled_images_mean` | 50.06 pp | ≥30 pp | Pass |
| `correct_minus_shuffled_questions_mean` | 42.00 pp | ≥30 pp | Pass |

Values are rounded for display; criteria use the unrounded report values. Triangle
training and validation now pass, along with all three dependence controls.
Circle and square training/validation accuracy remain below their thresholds.
Blank-image accuracy is 25.00%, mean shuffled-image accuracy 24.65%, and mean
shuffled-question accuracy 32.71%. These remain recipient-target controls.

### Comparison with the 2,880-update reference

Both runs use identical training and validation records. Validation changed as follows:

| Measure | 2,880 updates | 5,760 updates |
| --- | ---: | ---: |
| Overall accuracy | 66.84% | 74.71% |
| Circle accuracy | 52.08% | 65.28% |
| Square accuracy | 59.03% | 66.32% |
| Triangle accuracy | 89.41% | 92.53% |
| Circle/square pair accuracy | 32.29% | 56.77% |
| Identical circle/square predictions | 43.92% | 15.97% |
| All-three accuracy | 31.25% | 56.08% |
| Loss | 0.6160 | 0.5598 |

Overall validation accuracy improved by 7.87 percentage points. The 24.48-point
increase in correctly answering both circle/square questions, together with the
fall in identical predictions for their different targets, shows that the gain
includes the distinction this experiment was intended to investigate.

Training accuracy rose from 70.14% to 77.51%. Circle and square training accuracy
remain only 67.18% and 68.09%; triangle reaches 97.25%. Remaining errors therefore
are not solely a held-out transfer problem.

All four validation layouts improved, but unevenly:

| Geometry | 2,880 updates | 5,760 updates |
| --- | ---: | ---: |
| `validation:g000` | 71.99% | 79.40% |
| `validation:g001` | 67.59% | 80.09% |
| `validation:g002` | 72.92% | 83.10% |
| `validation:g003` | 54.86% | 56.25% |

The fourth layout remains substantially harder than the other three.

### Trajectory and interpretation

Validation progress remained noisy. Accuracy peaked at 76.39% at step 5,472,
then finished at **74.71% at step 5,760**. The final checkpoint remains the official
result; the earlier peak is not substituted. Validation loss finished at 0.5598,
compared with 0.4608 at step 5,472.

Additional training improved both training fit and held-out circle/square
behavior using the unchanged architecture. This supports retaining the current
architecture as a working baseline; it does not uniquely identify the remaining
learning bottleneck or establish that further training will resolve it.

Five of nine direct-grounding criteria pass, but reliable grounding and
**Milestone 2 remain incomplete**. This is one seed on four validation geometries,
not broad layout-generalization evidence or a recurrence comparison. Preserve
this run and its fixed budget. Selecting the next controlled follow-up is the
next decision; no additional experiment or training extension is established here.
