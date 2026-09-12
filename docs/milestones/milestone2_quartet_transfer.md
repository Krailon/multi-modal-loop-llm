# Milestone 2 — Frozen quartet transfer

## Status and question

**Completed; perfect reproduction and weak transfer across arrangements.** See
[results](#results). The
[quartet-fit experiment](milestone2_quartet_fit.md#results) achieved perfect
training fit. How much of that solution transfers to complete balanced quartets
at other arrangements that this checkpoint never saw?

This is a descriptive frozen-model diagnostic. There are no new accuracy gates,
optimizer steps, checkpoint selection, or validation/test inference. These scenes
belong to the project's training-source corpus; they are unseen by this particular
checkpoint, not a new untouched project test split. Milestone 2 remains incomplete.

## Frozen reference and population

Use `milestone2_quartet_fit_artifacts.zip` (or its extracted directory). Audit the
pinned artifacts before inference:

- Final checkpoint SHA256: `3e18600b01fcf94514d7a2aed9a1e4bf4d5b21958bd208966e57dee056f1fee4`
- Fit manifest SHA256: `86bdb774cef97e6b76752f016edd2b73473e8c989b77a7e2cedcb4e295894180`
- Reference summary SHA256: `842b57c8edd9606db37fe49d8a8f02b1e6d487d914ffad3566de332e40d5fd9d`
- Reference predictions SHA256: `e9f4898240012e0dd73241ca2b95463a701ef9ac0ead71a240ff709b5e86a41a`

Verify recorded hashes, deterministic fit/source derivation, and saved prediction
labels and summaries. Use the final 2,160-update checkpoint only. Preserve its
model configuration, R=2, float32, batch size 32 and direct question tokenizer.
Frozen loading creates no optimizer and restores no training randomness.

Enumerate all source arrangements containing every shape/color identity with a
complete four-size family. Select all seven with origins different from the learned
arrangement, without inspecting predictions. Triangle size remains 8 everywhere.

| Population | Ordered object origins | Role |
| --- | --- | --- |
| `learned` | (1,4), (10,4), (20,4) | Reproduction control |
| `transfer_01` | (1,11), (10,11), (19,11) | Transfer |
| `transfer_02` | (1,22), (14,22), (23,22) | Transfer |
| `transfer_03` | (2,9), (12,9), (21,9) | Transfer |
| `transfer_04` | (2,19), (11,19), (22,19) | Transfer |
| `transfer_05` | (2,21), (12,21), (22,21) | Transfer |
| `transfer_06` | (3,7), (14,7), (23,7) | Transfer |
| `transfer_07` | (3,16), (12,16), (21,16) | Transfer |

Each arrangement contains six shape orders × 24 distinct-color assignments ×
four circle/square size combinations = **576 images / 1,728 questions / 144
families**. Preserve sorted shape/color identity order and size order 6/6, 6/8,
8/6, 8/8, then question order square/circle/triangle. This permits direct matching
of questions and targets across arrangements.

The transfer population totals **4,032 images / 12,096 questions / 1,008 families**.
Including the learned reproduction control gives **4,608 images / 13,824 questions**.
Use existing rendering and pixels, blank margins, horizontal gaps and fixed origins
within families. No new rendering relaxation is introduced. Verify distinct scene
identities and origin separation from learned and reserved validation/test layouts.

The local read-only preflight verified all eight populations against the pinned
reference archive, without loading a model. The transfer manifest SHA256 is
`bc88c065e17e7ba449c7d209bd9816f1d5384c106f51117a8159f3eb1d70eb03`.

## Evaluation and interpretation

Evaluate the learned control and all seven transfer arrangements on the same frozen
model. Compare all 1,728 learned prediction IDs against the recorded reference.
Any disagreement marks reproduction unsuccessful, appears prominently in the
summary, console, notebook and inspection preview, and must be investigated before
interpreting transfer scores. Preserve all artifacts even in that case. Loss is
reported but is not required to match bit-for-bit across runtimes.

Report every arrangement separately and a transfer-only aggregate, excluding the
learned control. Include overall/per-shape accuracy and loss, circle/square pair
accuracy per size condition, relative-size swaps and other-color selections, and
families retaining both circle/square answers correctly across all four sizes.
Stable wrong answers do not count as correctly invariant families.

Match current transfer predictions against the current learned-control predictions
by ordered shape/color identity, size condition and queried shape. Record prediction
changes and correctness transitions (correct→wrong, wrong→correct, both correct,
both wrong), plus per-question comparison records. Keep the saved-reference
reproduction comparison separate. Preserve raw predictions and population metadata
for independent label, denominator and family audits.

Results remain descriptive, with no new transfer thresholds. Report layout
variation rather than treating thousands of related questions as independent
layout samples. Differences in position, spacing and patch offsets coexist here;
this check does not isolate their causal effects. It cannot establish general
visual reasoning, a recurrence advantage, or Milestone 2 completion.

## Interfaces and Kaggle workflow

Commit/push and import [the notebook](../../notebooks/kaggle_milestone2_quartet_transfer.ipynb).
Enable GPU and internet, set a committed `REPO_REF` (SHA preferred), and attach the
quartet-fit archive as `REFERENCE_SOURCE`. Use fresh checkout/output paths and
preserve Kaggle's installed PyTorch. Evaluation uses one CUDA device.

`prepare_quartet_transfer(repo, root, source)` audits the reference and writes the
population manifest and protocol. `run_quartet_transfer(run)` invokes only frozen
evaluation. `archive_quartet_transfer(run)` verifies hashes and packages the result,
including a failed reproduction check if one occurs. Helpers live in
`multimodal_loop.eval.kaggle_quartet_transfer`.

`scripts/evaluate_quartet_transfer.py` accepts `--source`, `--output-dir` and
`--device` (CPU/CUDA). `--smoke` permits test fixtures with different identities
and counts and labels their reports; the research notebook never supplies it.
Neither a training command nor a split-selection option is added.

The transfer manifest is a separate `quartet_transfer` version-1 metadata record:
source hashes, selected source indices, origins, roles, ordering and family
membership. Existing fit manifest bytes and checkpoint formats remain unchanged.
Reference checkpoint bytes are verified before and after inference; temporary
checkpoint staging is removed when evaluation ends.

Download `milestone2_quartet_transfer_artifacts.zip` for review. It contains
population metadata, complete predictions, summaries, matched comparisons,
reference summary/hashes, inspection HTML, protocol, revision/runtime, hashes and
logs. Reference model weights remain in the original input archive. The completed
research archive was reviewed as recorded below.


## Results

The completed `milestone2_quartet_transfer_artifacts.zip` reproduces the learned
arrangement perfectly, while transfer accuracy is only **3,851/12,096 (31.84%)**.
Only **62/1,008 transfer families (6.15%)** retain both circle/square answers
correctly across all four sizes. This is weak transfer under the descriptive
protocol; no new transfer accuracy gate was applied.

### Provenance and audit

The archive records revision `d48afd99e737c918debfc514554ca517e2ac6484`, PyTorch
2.10.0+cu128, CUDA 12.8, one Tesla T4, Python 3.12.13 and two CPU threads.
Evaluation used the pinned final quartet-fit checkpoint, R=2, float32 and batch
size 32, with zero model updates. There was no validation or test inference.

Local review verified all 10 artifact hashes, rebuilt the populations from the
pinned reference corpus, and checked all 13,824 saved predictions against labels
and population membership. Per-arrangement and transfer-only aggregate metrics,
family correctness, and matched prediction comparisons were recomputed from the
saved records. No new model inference was performed during review.

- Transfer manifest SHA256: `bc88c065e17e7ba449c7d209bd9816f1d5384c106f51117a8159f3eb1d70eb03`
- Summary SHA256: `2d807409f12e378aff116ffb1a92c4df19de2cd86cd26783a02874e6c7a2d835`
- Predictions SHA256: `17ff66263223046dba7f2e21030079d3583361873e86d3decbd236096d8ad1a8`
- Matched comparisons SHA256: `455091970ccf9aaca26a2225e55f6197470e47e47fbff7a3a97cd0f845c32dcf`

Reference checkpoint and fit-corpus identities remain those pinned above. The
learned arrangement is excluded from every transfer aggregate. The seven transfer
arrangements were unseen by this checkpoint but belong to the project's broader
training-source corpus; they are not a new untouched test split.

### Reproduction and aggregate transfer

| Measure | Learned arrangement | Seven transfer arrangements |
| --- | ---: | ---: |
| Overall accuracy | 100.00% | 31.84% |
| Circle accuracy | 100.00% | 28.89% |
| Square accuracy | 100.00% | 39.68% |
| Triangle accuracy | 100.00% | 26.93% |
| Circle/square pair accuracy | 100.00% | 12.15% |
| Families correct across all four sizes | 100.00% | 6.15% |
| Loss | 0.000406125 | 4.930942 |

All 1,728 learned-arrangement answers remain correct, with **zero prediction-ID
disagreements** against the saved reference. Reproduction therefore passes.
There are no invalid predictions in the transfer aggregate. Correct answers on
the learned arrangement do not establish transfer to different positions.

### Results by arrangement

Each transfer arrangement contains 1,728 questions and 144 complete families.
Origin coordinates are listed in the protocol table above.

| Arrangement | Accuracy | Circle | Square | Triangle | Pair accuracy | Families correct across all sizes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transfer_01` | 43.81% | 43.40% | 50.52% | 37.50% | 27.78% | 27/144 |
| `transfer_02` | 23.90% | 23.96% | 28.30% | 19.44% | 2.78% | 1/144 |
| `transfer_03` | 32.29% | 28.47% | 43.92% | 24.48% | 10.42% | 7/144 |
| `transfer_04` | 35.36% | 35.59% | 44.27% | 26.22% | 24.13% | 21/144 |
| `transfer_05` | 27.66% | 24.48% | 29.34% | 29.17% | 2.95% | 0/144 |
| `transfer_06` | 28.76% | 24.31% | 32.47% | 29.51% | 5.38% | 1/144 |
| `transfer_07` | 31.08% | 22.05% | 48.96% | 22.22% | 11.63% | 5/144 |

Accuracy ranges from 23.90% to 43.81%; no transfer arrangement approaches the
learned control. Even the best arrangement retains correct answers across all
four sizes for only 27/144 families. Triangle accuracy also deteriorates across
arrangements, extending the transfer problem beyond circle/square confusion.

### Size conditions and matched changes

Each size condition contains 3,024 transfer questions and 1,008 images. Pair
accuracy requires both circle and square answers to be correct on the same image.

| Circle size / square size | Overall accuracy | Pair accuracy |
| --- | ---: | ---: |
| 6 / 6 | 29.50% | 10.02% |
| 6 / 8 | 33.76% | 12.00% |
| 8 / 6 | 30.06% | 11.51% |
| 8 / 8 | 34.03% | 15.08% |

Match each transferred question to the current learned-control question with the
same shape/color identities and sizes. Because every control answer is correct,
each changed prediction is a correct-to-wrong transition. There are no
wrong-to-correct or both-wrong transitions.

| Arrangement | Changed predictions / correct-to-wrong | Both correct |
| --- | ---: | ---: |
| `transfer_01` | 971 | 757 |
| `transfer_02` | 1,315 | 413 |
| `transfer_03` | 1,170 | 558 |
| `transfer_04` | 1,117 | 611 |
| `transfer_05` | 1,250 | 478 |
| `transfer_06` | 1,231 | 497 |
| `transfer_07` | 1,191 | 537 |

Across all 12,096 matches, **8,245 predictions (68.16%) change from
correct to wrong**; 3,851 remain correct. These comparisons are distinct from the
saved-reference reproduction check, which has no disagreements.

### Interpretation and next decision

The quartet-fit success remains reproducible: the unchanged model learned every
balanced training question at the original arrangement. That solution does not
transfer reliably across the seven other arrangements. Location-specific visual
learning is a possible explanation, not an established internal mechanism.
Position, spacing and patch offsets change together, so this experiment does not
isolate their causal contributions or establish an architecture limitation.

The proposed next decision is a small balanced multi-arrangement training
experiment with substantial repeated exposure, an explicit update budget and
presentations per example, followed by evaluation on other arrangements. Its
purpose would be to test whether the successful training-fit solution becomes
more transferable when learned across several arrangements. That follow-up is
not specified or run by this results record.

Milestone 2 remains incomplete. No recurrence advantage has been tested. Preserve
all prior protocols and artifacts, and reserve test inference for a later
authorized step.
