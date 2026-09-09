# Milestone 2 — Direct shape-grounding sanity experiment

## Status and purpose

**Protocol fixed; implementation ready for a Kaggle run. No research results yet.**

The [original baseline diagnosis](milestone2.md#frozen-model-diagnosis--completed)
found near-perfect triangle-anchor training accuracy but approximately 50%
accuracy where circle/square discrimination was required. Layout transfer also
failed. This experiment asks whether the same architecture can learn direct
shape selection without the additional neighbor-selection requirement.

Use **“What color is the square/circle/triangle?”** on the exact original
three-object images. Keep the subtle circle/square renderings: changing them now
would prevent testing whether their existing distinction can be learned.
Training starts from scratch, not from the relational or Milestone 1 checkpoint.

## Fixed protocol

| Setting | Value |
| --- | --- |
| Source manifest SHA256 | `c5a9102eb619bfb134af48177bc1d1c2a083ce9b98d18a116be817b218c78b6c` |
| Images / layouts | Original 2,304 / 16 training; 576 / 4 validation |
| QA order | Stored image order, then square, circle, triangle |
| QA counts | 6,912 training; 1,728 validation |
| Rendering | Unchanged 32×32 RGB, object sizes 6 and 8 |
| Model | Original `configs/relational_baseline.yaml`, 172,928 parameters |
| Recurrence | R=2 for all training and evaluation |
| Question / answer | Six question tokens; one color token |
| Vocabulary | Original 17 IDs; color IDs remain 6–9 |
| Initialization / shuffle seed | 0 |
| Batch / dtype / device | 32; float32; single CUDA device `cuda:0` |
| Optimizer | AdamW, lr 0.001, weight decay 0, betas (0.9,0.999), eps 1e-8 |
| Optimizer implementation | `foreach=False`, `fused=False` |
| Budget | Exactly 2,880 updates, 92,160 QA presentations |
| Traversal | 13 full passes plus 72 batches of pass 14 |
| Monitoring | Validation at steps 0, 288, …, 2,880 |
| Selected checkpoint | Final step 2,880, irrespective of validation history |
| Controls | Blank images; image and within-image question shuffles, seeds 0–4 |

Labels are derived independently from the named object's color, not from stored
relational answers. All three questions are color-balanced within every geometry.
Metadata is used only for supervision and inspection, never as model input.
Prefix masking and answer-only shifted loss follow the established training path.
Frozen evaluation forwards question tokens only and takes full-vocabulary argmax;
non-color tokens count as errors.

Per-pass training indices use `random.Random(f"0:train:{epoch}")`, with zero-based
passes and no discarded batches. A pass contains 216 batches. The partial last
pass is deliberately recorded; it is not described as a complete epoch.

The baseline and this experiment match optimizer updates and QA presentations,
not total compute: the new questions are shorter (six versus eleven tokens), and
each direct QA is revisited more often. These are limitations on causal comparisons.
No scheduler, early stopping, recurrence sweep, extra training, or test inference
is part of this protocol. The embedded source manifest still contains reserved
test metadata, but no direct test dataset is constructed or evaluated.

## Reports and diagnostic criteria

Evaluate all training and validation examples using the final frozen checkpoint.
Record overall accuracy, cross-entropy, invalid outputs, and breakdowns by queried
shape, geometry, target position, and queried-object size. Include all-three-correct
accuracy, circle/square pair accuracy, identical circle/square prediction frequency,
and target-to-predicted-position confusion. The three colors in each image are
distinct, so every direct question pair requires different answers.

For validation, replace only images or only question tokens while preserving the
original recipient targets, batch order, and sequence layout. Image donor indices
address image records; question donor indices address three slots within an image.
Shuffles use ordinary seeded permutations with self-pairings retained. Save every
permutation and its same-answer pairing fraction. Blank/image controls have a
25% balanced color reference; shuffled questions have natural coincidences and
should be interpreted against their recorded answer-match fractions.

Predeclared diagnostic criteria, inclusive on unrounded metrics:

- **Training-set learning:** at least 95% accuracy for each of the three shapes.
- **Held-out direct grounding:** at least 90% accuracy for each shape on validation,
  plus at least 30 percentage-point overall gaps against blank images, mean
  shuffled-image accuracy, and mean shuffled-question accuracy.

Report all nine criteria separately. All-three and circle/square pair results are
additional diagnostics, not extra gates. These criteria do not replace the original
relational gates or complete Milestone 2. A four-layout validation split provides
limited evidence of spatial generalization.

Save every correct-input train/validation prediction with its target, confidence,
target probability, loss, and inspection metadata. The HTML preview shows the
eight highest-confidence wrong questions per split, with all three questions for
each selected image. It is deliberately selected, not a representative error sample.

## Run on Kaggle

Import [the notebook](../../notebooks/kaggle_milestone2_shape_grounding.ipynb) from
GitHub after committing and pushing these changes. Enable GPU and internet; attach
`milestone2_baseline_artifacts.zip` or its extracted directory. Set `BASELINE_SOURCE`
and use a committed `REPO_REF` (a commit SHA is preferred). Run all cells in order.
The notebook verifies the source archive's original baseline identity and records
its own revision/runtime before starting fresh training. It preserves Kaggle's
installed PyTorch stack and uses one GPU.

Use a fresh `RUN_ROOT` and retain the resulting
`milestone2_shape_grounding_artifacts.zip`. It includes:

- `training/`: self-contained `last.pt`, exact source manifest, settings and history.
- `diagnosis/`: summary, controls, train/validation JSONL predictions, HTML preview.
- `provenance/`: fixed protocol, source hashes, revision/runtime, final report hashes.
- `logs/`: complete subprocess output.

Staged original artifacts are excluded. Original weights are never used to
initialize this run. Checkpoints include model/configuration, optimizer, random
states, task/tokenizer identity, manifest, pass/batch progress, and metric history.
They support frozen evaluation on CPU/CUDA; **training resume is not implemented**
for this increment. A training interruption requires a fresh output directory and
restart from seed 0. A completed checkpoint can be evaluated independently:

```bash
python scripts/evaluate_shape_grounding.py \
  --checkpoint outputs/shape_grounding/training/last.pt \
  --output-dir outputs/shape_grounding/recheck \
  --device cuda:0
```

The training entry point is `scripts/train_shape_grounding.py`, accepting the
source `--manifest`, `--model-config`, fresh `--output-dir`, `--device`,
`--max-steps`, and `--evaluation-interval`. The notebook pins research settings.
`--smoke` permits a small non-baseline manifest for local infrastructure checks;
smoke artifacts are not research evidence. CPU end-to-end tests exercise the same
training, checkpoint and report entry points at a reduced budget.

## Interpretation

Strong training and validation performance would show that direct shape grounding
is learnable under this budget and motivate investigating the added relational
requirement. It would not prove the original relational model recognized shapes.
Strong training with weak validation identifies a transfer gap. Training failure
leaves visual discrimination, word-to-shape binding, and optimization unresolved;
it does not prove the shapes are indistinguishable or the architecture incapable.

Any renderer, geometry-coverage, or training-budget follow-up gets its own recorded
protocol. Preserve this run and the original baseline. None of these fixed-R results
establishes a recurrence advantage or disadvantage.
