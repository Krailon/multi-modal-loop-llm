# Milestone 2 — Frozen quartet transfer

## Status and question

**Protocol specified; research evaluation has not yet run.** The
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
logs. Reference model weights remain in the original input archive. No research
transfer result is recorded yet; local work is limited to selection preflight,
correctness tests and smoke evaluation with test weights.
