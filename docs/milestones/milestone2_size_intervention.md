# Milestone 2 — Frozen size intervention

## Status and question

**Frozen evaluation completed on Kaggle and audited locally.**
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

The staged reference checkpoint is excluded; no new checkpoint is trained. The
completed archive has been reviewed; research results are recorded below.


## Results

The completed `milestone2_size_intervention_artifacts.zip` contains frozen
inference on the specified **384 matched validation images**, with three questions
in each of five conditions: **5,760 QA evaluations**. No parameters were trained.

### Provenance and audit

The recorded revision is
`0597ddcd9ef38a75929c28e43d7a44e656cfb4b6`. The run used PyTorch 2.10.0+cu128,
CUDA 12.8, one Tesla T4, Python 3.12.13, and two CPU threads, at R=2 and batch size 32.
The checkpoint remains the specified final 5,760-update training-budget model.

Local review verified artifact hashes, exact source manifest, eligibility,
original/variant metadata, every target and prediction classification, recomputed
metrics, and agreement with saved original predictions. All **1,152 original-scene
predictions matched their saved reference predictions**, with zero disagreements.
No new model inference or test evaluation was performed during local review.

- Frozen checkpoint SHA256: `2069297aabdff24489bf2be5ef2f0c5a53989f4daaea312cd55e51a72282f8bc`
- Manifest SHA256: `047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975`
- Eligibility SHA256: `663cf61c7f6b6db49c9d97c27e436f92c22c160113da006e488a475e7053b736`
- Variant metadata SHA256: `19a60f0bdd395d265f61fdc9fbe0392e7bc6f2bc2423109f32c5fb9f11e9b327`
- Summary SHA256: `f161c9dadac4d12b669ca9eeb2b6e12dd3527182e8deb27c99c5f52790616f04`
- Predictions SHA256: `0cad55b8db29388cc9378d8079d49c3974125b36e32ff737a25ac2063db7b469`

### Matched-condition performance

Every row below uses the same 384 images and 1,152 questions, including 384
questions per shape. The original control is this eligible subset, not the full
576-image validation set used in the training-budget report.

| Condition | Overall | Square | Circle | Triangle | Both circle/square correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 71.53% | 63.28% | 61.98% | 89.32% | 53.12% |
| Circle 6, square 6 | 70.83% | 64.58% | 58.85% | 89.06% | 50.26% |
| Circle 6, square 8 | 92.97% | 98.44% | 90.10% | 90.36% | 88.80% |
| Circle 8, square 6 | 46.44% | 21.61% | 23.70% | 94.01% | 13.28% |
| Circle 8, square 8 | 82.81% | 71.88% | 77.08% | 99.48% | 61.98% |

With the square larger than the circle, both answers are correct on 88.80% of
images. Reversing their sizes reduces that to 13.28%, despite unchanged shape
identities, colors, origins and correct answers. Equal-size variants are also
imperfect; reliable recognition has not been established.

### Answer changes under size reversal

The reversal compares circle 6/square 8 with circle 8/square 6 on the same images:

- Square predictions change on 295/384 questions. Of these, **294 change from
  correct to the circle's color**, and one changes from correct to triangle color.
- Circle predictions change on 276/384 questions. **252 change from correct to
  the square's color**; the remaining transitions are retained in the report.
- Triangle predictions change on 41/384 questions between these two conditions,
  although the triangle itself is unchanged.

Across all four size variants, correct invariance is substantially weaker than
accuracy in the favorable unequal-size condition:

| Queried shape | Always correct | Invariant prediction | Invariant but incorrect | Prediction changes |
| --- | ---: | ---: | ---: | ---: |
| Square | 76/384 | 82/384 | 6/384 | 302/384 |
| Circle | 81/384 | 94/384 | 13/384 | 290/384 |
| Triangle | 335/384 | 337/384 | 2/384 | 47/384 |

Only **44/384 images (11.46%)** have both circle/square answers correct in all four
size variants. Stable incorrect answers are counted separately and are not
successful grounding. Triangle answers change across the four variants on
47/384 images, showing that the intervention can affect the unchanged object's
answer through the surrounding image context.

### Difficult-layout results

All 144 source images from `validation:g003` remain eligible. Each shape has 144
questions per condition:

| Condition | Overall | Square | Circle | Triangle |
| --- | ---: | ---: | ---: | ---: |
| Original | 56.25% | 50.00% | 42.36% | 76.39% |
| Circle 6, square 6 | 63.66% | 63.19% | 51.39% | 76.39% |
| Circle 6, square 8 | 88.66% | 100.00% | 85.42% | 80.56% |
| Circle 8, square 6 | 30.32% | 0.00% | 4.17% | 86.81% |
| Circle 8, square 8 | 87.27% | 77.78% | 84.72% | 99.31% |

The favorable relative-size condition yields 100% square accuracy and 85.42%
circle accuracy here; reversing sizes yields 0% and 4.17%. Triangle accuracy rises
from 76.39% on original scenes to 99.31% when both other shapes are size 8.
Consequently, the earlier triangle deficit cannot be explained solely by the
triangle's own pixels or position. This does not identify the interaction responsible.

### Interpretation and next decision

The intervention establishes strong size-dependent behavior and strengthens the
hypothesis that this model partly associates square with larger and circle with
smaller. It does not uniquely identify an internal rule: size changes also alter
area, contours, centers, patch coverage and, in some conditions, canvas-edge
contact. The matched population prevents changing sample membership between
conditions, but exclusions limit generalization to the full validation set.
These remain observations from one frozen model and four source layouts.

A controlled training-data follow-up with explicitly matched size variants is
recommended next. The unequal-size categories were already equally frequent in
training, so merely rebalancing those categories would not address the observed
problem. Systematic size variation within matched arrangements is the proposed
question; rendering constraints, held-out separation and the fixed training
budget must be specified before running it. No new training protocol or extension
is established by this report.

Preserve the current architecture and all reference artifacts for that comparison.
There are no new pass/fail gates, no recurrence-advantage claim, and no test-split
inference. **Milestone 2 remains incomplete.**

The subsequent [matched-size training protocol](milestone2_matched_size_training.md)
now specifies this follow-up with margins and a unique union of original and
variant training scenes. It has not yet been run; the frozen results above remain
unchanged.
