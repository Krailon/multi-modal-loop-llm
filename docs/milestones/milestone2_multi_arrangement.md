# Milestone 2 — Four-arrangement quartet training

## Status and question

**Completed; perfect training fit with improved but uneven transfer.** See
[results](#results). The
[quartet-fit run](milestone2_quartet_fit.md#results) achieved perfect training fit,
but its [frozen transfer](milestone2_quartet_transfer.md#results) was weak. Can
learning across several balanced arrangements, with substantial repeated exposure,
produce a more transferable solution using the unchanged model?

This is exploratory development. Previous arrangement scores are already known.
Use the fixed alternating order below, not a ranking by performance. Preserve the
project's reserved validation/test populations; neither is evaluated here.
Milestone 2 remains incomplete, and no recurrence advantage is being tested.

## Fixed corpus and arrangement assignment

Use the audited quartet-fit and quartet-transfer archives. Pin the fit manifest
SHA256 `86bdb774cef97e6b76752f016edd2b73473e8c989b77a7e2cedcb4e295894180`
and the recorded single-arrangement checkpoint identity
`3e18600b01fcf94514d7a2aed9a1e4bf4d5b21958bd208966e57dee056f1fee4`.
The reference weights never initialize training and are not re-evaluated.

The transfer summary SHA256 is
`2d807409f12e378aff116ffb1a92c4df19de2cd86cd26783a02874e6c7a2d835`, and its
prediction SHA256 is
`17ff66263223046dba7f2e21030079d3583361873e86d3decbd236096d8ad1a8`.
Verify source derivation, identities, saved prediction labels and successful
reference reproduction before using the comparisons.

Assign alternating entries in the established coordinate-sorted order. Names
retain their historical identifiers; `transfer_02`, for example, is now in the
new model's training population.

| Arrangement | Ordered origins | New role |
| --- | --- | --- |
| `learned` | (1,4), (10,4), (20,4) | Training |
| `transfer_01` | (1,11), (10,11), (19,11) | Final transfer |
| `transfer_02` | (1,22), (14,22), (23,22) | Training |
| `transfer_03` | (2,9), (12,9), (21,9) | Final transfer |
| `transfer_04` | (2,19), (11,19), (22,19) | Training |
| `transfer_05` | (2,21), (12,21), (22,21) | Final transfer |
| `transfer_06` | (3,7), (14,7), (23,7) | Training |
| `transfer_07` | (3,16), (12,16), (21,16) | Final transfer |

Preserve all 144 complete families per arrangement: six shape orders × 24
distinct-color assignments, with circle/square sizes 6/6, 6/8, 8/6, 8/8 and
triangle size 8. Reuse rendering, margins, gaps and within-family fixed origins.
Each arrangement has 576 images / 1,728 questions. Both the training and transfer
populations therefore contain **2,304 images / 6,912 questions / 576 families**.

Concatenate training arrangements in the table order, retaining each one's sorted
shape/color identities, fixed size-condition order and square/circle/triangle
question order. Record source indices, roles, families and balance in the new
manifest. Verify unique images and distinct origins across training, transfer
and reserved validation/test layouts. Scene metadata remains supervision and
inspection information, never model input.

The local preflight verified the pinned archives, assignments and exact 40-visit
exposure counts without loading or training a research model. The derived manifest
SHA256 is `ce16b936ce4ffcecb231794d9541a3c9d39461fef861c9e02349f9e4d68de8c7`.

## Training and computational budget

Initialize fresh seed-0 weights on CPU, then move to CPU/CUDA. Keep the existing
`configs/relational_baseline.yaml`: 172,928 parameters, 32×32 RGB, direct 8×8
patches, width 64, four heads, one block in each prelude/core/coda, R=2, dropout 0,
float32. Keep direct six-token questions, batch size 32, answer-only loss and
AdamW lr 0.001, weight decay 0, betas (0.9,0.999), eps 1e-8,
`foreach=False`, `fused=False`.

Uniformly shuffle all 6,912 training QAs each pass using
`random.Random(f"0:train:{epoch}")`. Do not batch or weight by family or arrangement.

| Budget measure | Fixed value |
| --- | ---: |
| Batches per complete pass | 216 |
| Complete passes | 40 |
| Updates | 8,640 |
| QA presentations | 276,480 |
| Presentations per training QA | Exactly 40 |
| Training-fit monitoring | Step 0, then every 216 updates |

Monitor the entire training population only, giving 41 history records. Persist
monitoring under `training_fit` and interval optimization metrics under `train`.
Do not monitor transfer arrangements during training. No resume, early stopping,
budget extension, learning-rate change or best-checkpoint selection. Evaluate
only the final 8,640-update checkpoint for the research comparison.

The budget preserves the single-arrangement run's 40 presentations per example
but uses **fourfold updates and total QA presentations**. Monitoring each larger
training population also adds evaluation cost. Improvement would support this
combined training setup; it would not isolate diversity from additional updates.

## Final assessment and reference comparisons

Evaluate all eight arrangements once at the final checkpoint: 13,824 questions.
Report every arrangement separately and distinct training/transfer aggregates.
For **each of the four training arrangements**, require:

- ≥99% accuracy for each queried shape (at least 571/576 correct per shape).
- ≥95% of families with both circle/square answers correct across all four sizes
  (at least 137/144 families).

All 16 checks must pass for the training-fit assessment to pass. An aggregate
score cannot conceal an individual arrangement's failure. Evaluate thresholds
without rounding. Stable wrong answers do not count as correct invariance.
These checks do not replace earlier protocols or milestone-completion gates.

Transfer remains descriptive, with no new accuracy thresholds. Record loss,
overall/per-shape accuracy, pair accuracy per size condition, relative-size swaps,
and families correct across all four sizes. Preserve raw predictions and preview
confident errors for training and transfer arrangements.

Compare the four final-transfer arrangements against their exact saved
single-arrangement-model predictions. Recompute the reference aggregate on
**these same four arrangements / 6,912 questions**, excluding both the former
seven-arrangement aggregate and the new training arrangements. Report overall,
per-shape, loss, pair and family differences, plus per-question matched prediction
and correctness transitions. No reference-model inference is needed.

The four transfer arrangements are unseen by the new model's training, but are
part of the project's previously evaluated training-source corpus. They are not
an untouched test split. Any conclusion remains limited to one seed and these
arrangements; changes in arrangement diversity and computation are confounded.

## Interfaces and Kaggle workflow

Commit/push and import [the notebook](../../notebooks/kaggle_milestone2_multi_arrangement.ipynb).
Enable GPU and internet, set a committed `REPO_REF` (SHA preferred), and attach:

- `milestone2_quartet_fit_artifacts.zip` as `FIT_SOURCE`.
- `milestone2_quartet_transfer_artifacts.zip` as `TRANSFER_SOURCE`.

ZIPs or extracted directories are accepted. Use fresh checkout/output paths and
preserve Kaggle's installed PyTorch. Research training uses one CUDA device.

`prepare_multi_arrangement(repo, root, fit_source, transfer_source)` audits both
references and writes the manifest, derivation, exposure counts and protocol.
`run_multi_arrangement(run)` invokes fresh training followed by final diagnostics.
`archive_multi_arrangement(run)` packages verified artifacts. Helpers live in
`multimodal_loop.eval.kaggle_multi_arrangement`.

`scripts/train_multi_arrangement.py` accepts `--manifest`, `--model-config`,
`--output-dir`, `--device`, `--max-steps` and `--evaluation-interval`. Research
settings are fixed and enforced. `--smoke` permits explicitly marked test fixtures
and budgets. `scripts/evaluate_multi_arrangement.py` accepts `--checkpoint`,
`--reference-source`, `--output-dir` and `--device`; smoke status follows the
checkpoint. Neither entry point exposes the project's validation/test split.

The distinct manifest/checkpoint kind is `multi_arrangement_quartet_fit`, version
1. Preserve configuration, model/optimizer state, tokenizer, corpus identity,
history, progress and randomness state. Frozen loading does not restore training
randomness or support resume. Existing manifests and checkpoint formats remain
unchanged.

Download `milestone2_multi_arrangement_artifacts.zip` for review. It contains the
fresh checkpoint, data and reference provenance, protocol, settings, exposure
counts, history, final predictions, per-arrangement/aggregate metrics, matched
comparisons, inspection HTML, hashes, logs and runtime/revision records. Reference
weights are excluded. The completed archive was reviewed as recorded below.


## Results

The completed `milestone2_multi_arrangement_artifacts.zip` achieves perfect fit
on all four training arrangements. Transfer improves on the identical four
reserved arrangements, but most of the gain is concentrated in `transfer_01`.
Reliable transfer across positions remains unresolved.

### Provenance and audit

The archive reports clean revision `d44153fb41055e63cdc6833c14041ed129f7e711`, PyTorch
2.10.0+cu128, CUDA 12.8, one Tesla T4, float32 and two CPU threads. Fresh seed-0
training completed **8,640 updates / 276,480 QA presentations / 40 passes** with
the specified model, R=2 and optimizer. Every training QA received 40 presentations.

Local review verified all 18 artifact hashes, regenerated the fixed corpus,
checked exposure counts and checkpoint progress, and validated all 13,824 saved
predictions against their arrangement labels. Per-arrangement metrics, training
and transfer aggregates, training criteria and matched reference comparisons were
recomputed from saved records. No new model inference was performed during
review. No project validation/test inference was performed by this experiment.

- Manifest SHA256: `ce16b936ce4ffcecb231794d9541a3c9d39461fef861c9e02349f9e4d68de8c7`
- Final checkpoint SHA256: `4e2adf98a2fd7ca2ba458c1a99a05973c5744b1db2a0b57e1810af2c56b8be61`
- Summary SHA256: `367b04f0c48120eea105b90d1664906e1a38890849dcc732397177555aee1970`
- Predictions SHA256: `1d616e65d1721b726e287d19b9d6b08040cd28860958e962a842b978c55239c8`
- Comparison SHA256: `6ebd8a3cdec894c803cc149d3dcc902da9d5a466cd5c526a0851f70475ab8ade`

### Training fit and trajectory

All **6,912/6,912 training questions** are correct. Every training arrangement
has 100% accuracy for square, circle and triangle, and all **576/576 families**
retain both circle/square answers correctly across all four sizes. All **16/16
training-fit criteria pass**, assessed per arrangement rather than only in aggregate.
Final training loss is 0.0000156083.

The first recorded perfect training evaluation occurs at **update 3,888 / pass
18**, and accuracy remains 100% at every subsequent recorded evaluation through
pass 40. Monitoring occurred once per pass, so this identifies the first recorded
perfect score, not the exact update at which every answer became correct. The
final checkpoint remains the specified result; no earlier checkpoint was selected.

### Same-population transfer comparison

Both columns below refer to **transfer_01/03/05/07: 6,912 questions and 576
families**. The reference is the saved single-arrangement model evaluated on these
same four arrangements. Its accuracy is therefore **33.71%**, not the **31.84%**
previously reported across all seven transfer arrangements. Neither aggregate
includes the new training arrangements.

| Measure | Single-arrangement reference | Four-arrangement model |
| --- | ---: | ---: |
| Overall accuracy | 33.71% | 44.34% |
| Circle accuracy | 29.60% | 45.96% |
| Square accuracy | 43.19% | 49.65% |
| Triangle accuracy | 28.34% | 37.41% |
| Circle/square pair accuracy | 13.19% | 27.47% |
| Families correct across all sizes | 6.77% | 13.54% |
| Loss | 4.795463 | 5.748286 |

Transfer accuracy improves from **2,330/6,912 to 3,065/6,912**, a gain of **10.63
percentage points**. Correct-family counts rise from **39/576 to 78/576**. Loss
nevertheless worsens from 4.80 to 5.75: higher answer accuracy did not improve the
average negative log probability of the correct answer. There are no transfer
accuracy gates in this experiment.

### Variation across transfer arrangements

Each arrangement contains 1,728 questions and 144 families.

| Arrangement | Reference accuracy | New accuracy | New circle | New square | New triangle | New correct families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transfer_01` | 43.81% | 78.99% | 68.40% | 81.42% | 87.15% | 72/144 |
| `transfer_03` | 32.29% | 25.93% | 31.25% | 36.28% | 10.24% | 2/144 |
| `transfer_05` | 27.66% | 29.46% | 45.31% | 31.60% | 11.46% | 0/144 |
| `transfer_07` | 31.08% | 43.00% | 38.89% | 49.31% | 40.80% | 4/144 |

**72 of the 78 correctly invariant transfer families occur in `transfer_01`.**
Across the other three arrangements, only **6/432 families (1.39%)** retain both
answers correctly across all four sizes. `transfer_03` regresses, and triangle
accuracy is particularly low on `transfer_03` and `transfer_05`. The aggregate
improvement therefore does not establish broad position transfer.

### Matched correctness changes

Each row compares identical questions, colors and size conditions in the same
arrangement between the reference and new models. Prediction changes can include
wrong-to-wrong changes and are distinct from correctness transitions.

| Arrangement | Prediction changes | Correct→wrong | Wrong→correct | Both correct | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transfer_01` | 981 | 125 | 733 | 632 | 238 |
| `transfer_03` | 1,048 | 373 | 263 | 185 | 907 |
| `transfer_05` | 1,213 | 327 | 358 | 151 | 892 |
| `transfer_07` | 1,104 | 244 | 450 | 293 | 741 |

### Interpretation and next decision

The unchanged architecture and optimizer can fit several balanced arrangements
with repeated exposure. Transfer improves, substantially at one arrangement,
but remains uneven and unreliable elsewhere. Location-specific visual learning
remains a possible explanation, not an established internal mechanism.

This run uses fourfold updates and QA presentations relative to the
single-arrangement fit run. Diversity and additional computation therefore remain
confounded; improvement cannot be attributed to diversity alone. Prior scores
were known before this exploratory experiment, and the transfer population is
part of the project's training-source corpus, not an untouched test split.

The proposed next decision is work targeting broader position transfer with an
explicit data and compute budget. Further training on these same four arrangements
alone would be less informative given their sustained perfect fit. No follow-up
experiment is specified or run by this results record.

Milestone 2 remains incomplete. No recurrence advantage has been tested. Preserve
all prior protocols and artifacts, and reserve test inference for a later
authorized step.
