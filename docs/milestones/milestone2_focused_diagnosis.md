# Milestone 2 — Focused size and layout diagnosis

## Status and evidence

**Completed locally using saved predictions only.** This compares the
[2,880-update geometry-diversity run](milestone2_geometry_diversity.md#results)
with the [5,760-update budget run](milestone2_training_budget.md#results).
No model checkpoints were deserialized, no model inference or training was run,
and no test predictions were accessed. Both archives contain the same manifest.

The analyzer checked manifest/summary hashes, every train/validation row's index,
question, target, prediction decoding, shape, geometry, size and position, and
reconciled grouped metrics and aggregate counts with the original reports.
Overall loss reconciliation permits float32 reduction differences (observed
absolute differences below 4e-9); labels, counts and grouped summaries match exactly.
Source archive and prediction-file hashes are retained in the generated JSON.

## Relative size separates strong and weak behavior

Each training unequal-size category contains **4,224 images**; the equal-size
category contains 9,984. Validation has **192 images in each category**. Each image
provides one circle question and one square question, with different target colors.

| Both questions correct | Train: 2,880 | Train: 5,760 | Validation: 2,880 | Validation: 5,760 |
| --- | ---: | ---: | ---: | ---: |
| Square smaller than circle | 3.60% | 15.58% | 6.77% | 24.48% |
| Equal size | 39.28% | 64.04% | 28.65% | 51.04% |
| Square larger than circle | 67.42% | 91.74% | 61.46% | 94.79% |

The asymmetry is not explained by simply having more training examples where the
square is larger. Additional training helped paired answers in every category,
but the smaller-square category remains particularly weak. Its individual
training accuracy changed from 19.22% to 23.08% for squares and from 31.72% to
27.70% for circles. Paired improvement does not mean every marginal improved.

### What the wrong answers select

For circle/square questions, classify predictions as correct, the other
circle/square shape's color, triangle color, absent color, or invalid token.
The following counts are from the **5,760-update** run when the square is smaller:

| Split / queried shape | Questions | Correct | Other shape's color | Triangle color | Swap / all questions | Swap / errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training / square | 4,224 | 975 | 3,241 | 8 | 76.73% | 99.75% |
| Training / circle | 4,224 | 1,170 | 3,017 | 37 | 71.43% | 98.79% |
| Validation / square | 192 | 62 | 129 | 1 | 67.19% | 99.23% |
| Validation / circle | 192 | 69 | 122 | 1 | 63.54% | 99.19% |

There are no absent-color or invalid-token predictions in these groups. Errors
almost exclusively select the other circle/square object's color. This is
consistent with a learned association between square/larger and circle/smaller,
not proof of a particular internal representation or algorithm. Relative size,
spatial arrangement and visual appearance remain coupled in this observational
analysis. Equal-sized shapes are also far from reliably solved.

### Changes on the same questions

Across all 1,728 validation questions, 257 changed from wrong to correct and 121
from correct to wrong; 316 remained wrong and 1,034 remained correct. Their net
change is the recorded gain of 136 correct answers. On training, the corresponding
counts are 8,228, 4,155, 8,283 and 34,630.

Within smaller-square validation cases, square questions had 39 corrections and
33 regressions; circle questions had 27 corrections and 22 regressions. Most of
the overall progress occurred outside this weak category. The JSON retains these
transitions by relative size and queried shape for both splits.

## The difficult layout has an additional position-sensitive failure

`validation:g003` places objects at `(left, top, size)`:
`(4,23,8)`, `(14,24,6)`, `(23,24,6)`. Each queried-shape/position cell has **48
questions**, varying colors and assignments of the other shapes.

| Queried shape | Left: size 8, patch offset (4,7) | Middle: size 6, offset (6,0) | Right: size 6, offset (7,0) |
| --- | ---: | ---: | ---: |
| Circle | 0 / 48 (0.00%) | 33 / 48 (68.75%) | 28 / 48 (58.33%) |
| Square | 48 / 48 (100.00%) | 17 / 48 (35.42%) | 7 / 48 (14.58%) |
| Triangle | 15 / 48 (31.25%) | 47 / 48 (97.92%) | 48 / 48 (100.00%) |

Offsets are the object's origin modulo the 8-pixel patch size, not the patch
index. The report includes the corresponding joint groups for all four
validation geometries and all training geometries at both checkpoints.

All 48 left-position circle questions choose the square's color. For the
left-position triangle, 22 answers select the circle, 11 the square, and 15 the
triangle correctly. Triangle's decline shows that this layout's deficit cannot
be reduced to swapping the two similar shapes alone.

Position, absolute coordinates, size and patch offsets are confounded within a
fixed geometry. These counts do not demonstrate that patch boundaries cause the
failure, or that the same offset fails across other placements. They motivate a
controlled intervention rather than an architectural conclusion.

## Matched examples and reproduction

The [self-contained rendered preview](milestone2_focused_diagnosis.html) contains
up to eight selected errors with matched successes where available. Selection is
deterministic by example index, alternating validation and training, with at most
one error per geometry/shape-assignment/query group. Successes share the split,
geometry, shape assignment and queried shape but can differ in color assignment.
Missing matching successes are stated explicitly. These examples are illustrative,
not a representative sample; the complete counts above carry the conclusions.

Reproduce the full JSON and preview locally from the preserved archives:

```bash
python scripts/diagnose_grounding_errors.py \
  --reference milestone2_geometry_diversity_artifacts.zip \
  --current milestone2_training_budget_artifacts.zip \
  --output-dir outputs/milestone2_focused_diagnosis
```

Use a fresh output directory. The analyzer reads ZIP members without extraction;
only train/validation prediction files are analyzed. `report.json` contains source
hashes, relative-size categories, prediction classifications, transitions, joint
layout groups and preview indices. `inspection.html` renders the actual synthetic
scenes with the existing renderer, without invoking a model. Outputs are ignored
by Git; the compact interpretation and preview are retained here.

## Next question

A useful next controlled question is whether predictions follow shape identity
when object sizes are changed while placements and colors are held fixed, with
all resulting scenes checked for valid bounds and non-overlap. Separately varying
placement would help investigate the difficult layout. Such interventions could
separate factors that the saved observations cannot isolate.

This is a recommendation, not a new evaluation or training protocol. Preserve the
current architecture, training budgets, artifacts and reserved test split while
specifying the next step. Milestone 2 remains incomplete; this diagnosis neither
establishes a recurrence advantage nor authorizes further training or inference.

The subsequent [size-intervention protocol](milestone2_size_intervention.md) now
specifies this frozen-model evaluation, including edge-contact eligibility. Its
[completed results](milestone2_size_intervention.md#results) establish strong
size-dependent behavior. This saved-prediction diagnosis remains unchanged.
