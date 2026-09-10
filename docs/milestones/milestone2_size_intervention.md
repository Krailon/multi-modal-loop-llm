# Milestone 2 — Frozen size intervention

## Status and question

**Protocol specified and locally checked; research inference has not yet run.**
The [focused diagnosis](milestone2_focused_diagnosis.md) found strong relative-size
dependence in circle/square errors. Does changing their sizes at fixed object
origins and colors systematically change the frozen model's answers?

Use only the final **5,760-update training-budget checkpoint**, SHA256
`2069297aabdff24489bf2be5ef2f0c5a53989f4daaea312cd55e51a72282f8bc`.
Preserve its parameters, R=2, float32 and direct-question vocabulary. This is
frozen evaluation, not additional training or a new milestone acceptance gate.

## Matched scenes and eligibility

Take validation scenes from the unchanged manifest, SHA256
`047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975`.
Keep each object's shape, color, left/top origin, and the triangle's size fixed.
Ask the same square, circle and triangle color questions in all conditions.
All correct answers remain unchanged.

| Condition | Circle size | Square size |
| --- | ---: | ---: |
| Original | As stored | As stored |
| `circle6_square6` | 6 | 6 |
| `circle6_square8` | 6 | 8 |
| `circle8_square6` | 8 | 6 |
| `circle8_square8` | 8 | 8 |

Determine eligibility before inference: every variant must remain fully visible
inside the 32×32 canvas with at least one blank column between adjacent bounding
boxes. Exclude a source image from **all** conditions if any variant fails; record
the failing conditions and reasons. Never clip an object or compare different
populations across size conditions.

As explicitly agreed, this diagnostic allows canvas-edge contact and relaxes
vertical center alignment. Original corpus validation remains unchanged. A
separate diagnostic scene type reuses the existing single-object renderer on a
34×34 canvas, then crops background to 32×32 after validating full visibility.
This preserves shape masks and colors; it does not crop object pixels.

| Source geometry | Eligible images | Excluded images |
| --- | ---: | ---: |
| `validation:g000` | 48 | 96 |
| `validation:g001` | 144 | 0 |
| `validation:g002` | 48 | 96 |
| `validation:g003` | 144 | 0 |
| Total | 384 | 192 |

The local preflight confirmed these counts and pixel-exact reconstruction of all
384 original scenes. Canvas-edge contact occurs in 0 original images, 0 both-size-6
images, 144 images in each unequal-size condition, and 240 both-size-8 images.
These are pre-inference geometry counts, not model results. Exclusions change
layout/shape-assignment coverage; report each layout with its denominator.

## Evaluation and reports

Evaluate all three questions on all five conditions: **5,760 QA evaluations**.
Use batch size 32, no shuffling, a single CUDA device, evaluation mode and no
gradients. Forward only images and six question tokens; target tokens and scene
metadata never enter model inputs. Use existing frozen checkpoint loading and
color-prediction utilities. Test examples are not evaluated.

Record every prediction, target, loss, confidence, source index, condition,
source geometry, predicted shape and canvas-edge exposure. Report:

- Accuracy and counts by condition, queried shape and source layout; circle/square
  pair accuracy and selections of the other circle/square shape's color.
- Prediction and category changes from `circle6_square8` to `circle8_square6`.
- Per-shape predictions invariant across the four size variants, separating
  always-correct predictions from invariant incorrect predictions.
- The fraction of images where **both** circle/square questions remain correct
  across all four size variants.
- Triangle accuracy and answer changes as a diagnostic of effects on the unchanged
  object, and the difficult layout's condition-specific results.

Rerun originals on the same eligible population and compare their predictions
with saved reference rows. Report every disagreement explicitly; device/runtime
and batch-composition differences can affect numerical results. Do not substitute
saved predictions for the original control. Verify the checkpoint hash remains
unchanged before and after evaluation.

The preview shows the first eligible source image per geometry across all five
conditions, selected independently of outcomes. No blank-image, shuffled-question,
training, placement sweep or recurrence sweep is added to this experiment.

## Interpretation limits

A systematic change under size reversal establishes sensitivity to this
intervention and strengthens the size-shortcut hypothesis. It does not uniquely
identify an internal computation: resizing changes area, contours, bounding-box
centers, patch coverage and sometimes edge contact. Triangle size and origins
remain fixed, but the other objects' changed pixels may affect its answer.

There are no new pass/fail thresholds. This is one frozen model on a selected
subset of four validation layouts, not a broad generalization or recurrence
comparison. Stable wrong answers are not evidence of successful grounding.
Milestone 2 remains incomplete. Any subsequent training change or placement
intervention requires a separate specification.

## Kaggle workflow

After committing and pushing, import
[the notebook](../../notebooks/kaggle_milestone2_size_intervention.ipynb) from GitHub.
Enable GPU and internet, attach `milestone2_training_budget_artifacts.zip`, set
`REFERENCE_SOURCE` to the ZIP or extracted run directory, and select a committed
`REPO_REF` (commit SHA preferred). Use fresh checkout/output paths and run all cells.
The notebook preserves Kaggle's PyTorch installation and uses `cuda:0`.

The preparation step audits the pinned checkpoint, settings, manifest and reports
and writes eligibility and variant metadata before inference. The helper API is
`prepare_size_intervention(repo, root, source)`, `run_size_intervention(run)` and
`archive_size_intervention(run)` in `multimodal_loop.eval.kaggle_size_intervention`.
The evaluation command accepts `--reference-dir`, `--output-dir` and `--device`:

```bash
python scripts/evaluate_size_intervention.py \
  --reference-dir /path/to/milestone2_training_budget \
  --output-dir outputs/size_intervention_evaluation \
  --device cuda:0
```

The default notebook archive is `milestone2_size_intervention_artifacts.zip`:

- `data/`: source manifest, every eligibility decision, original/variant metadata.
- `diagnosis/`: summary, all predictions, original-reference agreement and HTML preview.
- `provenance/`: fixed protocol, reference report/hashes, runtime/revision and final hashes.
- `logs/`: streamed evaluation output.

The staged reference checkpoint is excluded; no new checkpoint is trained. Bring
this archive back for review. Local CPU fixtures and mocked orchestration validate
implementation only; no research accuracy results have been recorded yet.
