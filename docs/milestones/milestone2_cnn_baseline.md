# Milestone 2 — Conventional CNN grounding baseline

## Status and decision

**Specified and implemented; research run pending.** This is a small, independent
vision baseline for the current direct shape-to-color task, trained from scratch
on the exact [four-arrangement corpus](milestone2_multi_arrangement.md). It does
not change the recurrent transformer or establish relational reasoning.

The preceding run reached perfect training fit and 44.34% transfer accuracy, but
72/78 correctly invariant transfer families came from one arrangement. Before
further dataset adjustments, test whether conventional local visual processing
learns a more transferable solution on these same examples.

## Focused literature review

This is a targeted review of visual inductive biases, spatial pooling, training
recipes, and question conditioning, not a comprehensive survey of modern VLMs.
The selected architecture is our small control, not a reproduction of a paper.

| Primary source | Finding relevant here | Decision and limitation |
| --- | --- | --- |
| Dosovitskiy et al., [An Image Is Worth 16x16 Words](https://arxiv.org/html/2010.11929v2), ICLR 2021, §3.1 and §4.3 | ViTs have fewer built-in locality and translation assumptions; data scale affects their generalization relative to CNNs. | Limited spatial coverage is a plausible difficulty for our patch model. ImageNet-scale evidence does not identify the cause of our errors. |
| Touvron et al., [DeiT](https://proceedings.mlr.press/v139/touvron21a.html), ICML 2021 | Strong convolution-free transformers can be trained without external pretraining through a carefully designed recipe. | Do not conclude that pure transformers, small models, or training from scratch are inherently unsuitable. |
| Xiao et al., [Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881), NeurIPS 2021 | A convolutional stem improved ViT optimization stability and accuracy in their settings. | Early visual processing is a candidate follow-up if the independent CNN succeeds; it is not a guaranteed fix. |
| Lin et al., [Network in Network](https://arxiv.org/html/1312.4400v3), ICLR 2014, §3.2 | Global spatial pooling replaces a location-specific flattened classifier and promotes robustness to translations. | Pool visual features for direct attribute lookup. Our subsequent question-conditioned head differs from NIN's classifier. |
| Zhang, [Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html), ICML 2019 | Downsampling can introduce translation sensitivity even in CNNs. | Use stride-one convolutions and preserve full resolution until global pooling. Borders still preclude a blanket exact-invariance claim. |
| Perez et al., [FiLM](https://arxiv.org/html/1709.07871v2), AAAI 2018, §2.2 and §4 | Question-conditioned visual reasoning works with learned convolutional processing, including raw-pixel training. | A conventional vision/question baseline is credible. Defer FiLM and spatial reasoning machinery for this direct-grounding control. |

Our inference is that visual representation and training coverage both remain
plausible contributors. The 6–8-pixel shapes are comparable to the transformer's
8×8 patches; translation changes within-patch coordinates and contour fragments.
The shared patch projection is not inherently pixel-translation equivariant,
and learned absolute positions permit location-specific solutions. Thousands of
color/question combinations over four arrangements do not supply thousands of
spatial arrangements. None of this establishes an internal failure mechanism.

## Model and inputs

Use `configs/cnn_baseline.yaml`, implemented by `CNNBaseline` with a separate
`CNNConfig` in `multimodal_loop.model.cnn_baseline`:

- Existing float32 RGB `[B,3,32,32]` images, with unchanged rendering/pixel values.
- Five 3×3 convolutions, channels `3→32→64→64→64→64`; stride 1, zero padding 1,
  bias enabled, ReLU after every layer. No intermediate downsampling.
- Global average pooling gives 64 visual features. Receptive field: 11×11.
- Learned `Embedding(17,32)` averaged over all six existing question tokens.
- Concatenate the 64 image and 32 question features; apply
  `Linear(96,128) → ReLU → Linear(128,17)`.
- Standard PyTorch initialization, fresh seed 0 on CPU before device transfer;
  no dropout, normalization layers, augmentation, or additional regularization.

There are **145,329 parameters**. `forward(images, question_ids)` returns `[B,17]`
raw logits. Mean cross-entropy supervises the single external answer target.
Prediction uses all 17 tokens; non-color outputs are incorrect. No scene metadata,
coordinates, object crops, shape-specific routing, or answer tokens enter forward.
The batch collator reads only image, question text, and external answer labels.

Question averaging is sufficient for these three fixed templates; it is not a
language encoder for general or relational questions. Global pooling is appropriate
for direct attributes, but this model is not a proposed replacement for the
project's eventual spatial and recurrent reasoning architecture. Borders and
neighboring objects may still affect its features.

## Corpus and budget

Read `milestone2_multi_arrangement_artifacts.zip` from the repository root.
Audit the pinned manifest, saved predictions, summary and reference checkpoint
identity without loading reference weights. The manifest SHA256 is
`ce16b936ce4ffcecb231794d9541a3c9d39461fef861c9e02349f9e4d68de8c7`.
The reference checkpoint SHA256 is
`4e2adf98a2fd7ca2ba458c1a99a05973c5744b1db2a0b57e1810af2c56b8be61`.
Source prediction and summary hashes are pinned in `eval/cnn_reference.py`.

Train on `learned/transfer_02/transfer_04/transfer_06`; assess final transfer on
`transfer_01/transfer_03/transfer_05/transfer_07`. Each population has **2,304 images,
6,912 QAs, 576 complete families**. Preserve all shape/color combinations, four
circle/square size conditions, triangle size 8, gaps, margins and fixed origins.
The manifest loader rederives populations and their separation. Project validation
and test populations are not exposed by these entry points.

| Budget measure | Fixed value |
| --- | ---: |
| Research runs | One, seed 0 |
| Batch size | 32 |
| Updates | 8,640 |
| Batches per pass | 216 |
| Complete passes / presentations per QA | 40 / exactly 40 |
| Total training QA presentations | 276,480 |
| Training-fit monitoring | Step 0, then every 216 updates; 41 records |
| Monitoring QA presentations | 283,392 |
| Final correct-image evaluation QA presentations | 13,824 |

Shuffle all training QAs each pass with `random.Random(f"0:train:{epoch}")`.
Use AdamW lr 0.001, weight decay 0, betas (0.9,0.999), eps 1e-8,
`foreach=False`, `fused=False`. Single CUDA device, float32. Monitor training fit
only. No resume, search, early stopping, budget extension, best-checkpoint
selection, or transfer-driven tuning. Assess only the final checkpoint.

This matches updates and data exposure, **not compute**. Convolution/linear forward
cost is **133,019,776 MACs per QA**, excluding activations, pooling and backward.
Record training-plus-monitoring wall time and peak allocated/reserved CUDA memory.
The smaller parameter count does not imply faster execution. The notebook reports
end-to-end subprocess timing separately through its logs/cell execution.

## Assessment and interpretation

Reuse the previous training criteria for **each** training arrangement:
≥99% accuracy for each queried shape (571/576) and ≥95% of families with both
circle/square answers correct across all sizes (137/144). All 16 checks must pass.
Thresholds are evaluated without rounding; stable wrong answers are not correct
invariance. These are training-fit criteria, not milestone completion gates.

Transfer remains descriptive, without new accuracy gates. Report every arrangement
and separate training/transfer aggregates: overall/per-shape accuracy and loss,
circle/square pairs, size-condition results, relative-size swaps, complete-family
correctness, and confident errors. Compare the exact saved **four-arrangement
transformer** predictions, whose matching transfer score is 44.34%, using matched
correctness changes. Do not substitute the old seven-arrangement aggregate or
retrain/re-evaluate reference weights.

- Strong fit and consistently better transfer support investigating the
  transformer's visual representation and training recipe.
- Strong fit with weak transfer leaves spatial coverage and representation unresolved.
- Failed fit makes this baseline inconclusive; it does not establish a defective
  dataset or an architectural limitation of the transformer.

This changes the whole model family, including question encoding and answer
computation. It cannot isolate patch size, positional embeddings, recurrence or
optimization. It is a single-seed exploratory control on previously evaluated
training-source arrangements, not an untouched test or a recurrence comparison.
Milestone 2 remains incomplete. Any next convolutional-stem intervention requires
a separate protocol.

## Kaggle workflow and artifacts

Commit/push the implementation and import
[the CNN notebook](../../notebooks/kaggle_milestone2_cnn_baseline.ipynb).
Enable GPU and internet, set `REPO_REF` to the committed revision, and run cells
in order. `SOURCE` defaults to
`Path(REPO_DIR) / "milestone2_multi_arrangement_artifacts.zip"`.
Use a fresh output directory. Installation preserves Kaggle's PyTorch.

Helpers `prepare_cnn_baseline`, `run_cnn_baseline`, and `archive_cnn_baseline` live
in `multimodal_loop.eval.kaggle_cnn_baseline`. Training and evaluation use
`scripts/train_cnn_baseline.py` and `scripts/evaluate_cnn_baseline.py`.
Research settings are enforced; explicit `--smoke` allows small fixtures/budgets
and labels the checkpoint and report accordingly. No validation/test flags exist.

The separate checkpoint kind is `cnn_direct_shape_color`, version 1. Save model
configuration/state, optimizer state, tokenizer, manifest content/hash, progress,
history and CPU/device randomness. Frozen loading preserves the caller's randomness;
training resume is unsupported. Existing transformer formats are unchanged.

Download `milestone2_cnn_baseline_artifacts.zip`. It contains data identities,
protocol, exposure counts, checkpoint, training history, runtime, raw predictions,
per-arrangement metrics, matched comparisons, inspection HTML, logs and hashes.
The notebook displays all transfer arrangements alongside their reference scores,
plus correct-family counts, rather than only an aggregate.

Local correctness checks cover input/target separation, gradients, deterministic
sampling, protocol rejection, frozen checkpoint predictions, reference audit and
metric compatibility. Research results will be recorded here after the Kaggle run.
