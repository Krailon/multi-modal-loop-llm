# AGENTS.md

## Project

This repository implements an experimental **multimodal recurrent-depth transformer** trained from scratch.

Read `README.md` before making architectural changes. It describes the research goals, hypotheses, milestones, and intended repository structure.

The current priority is **Milestone 1: synthetic vision and visual grounding**.

Do not prematurely optimize for large-scale training.

---

## Core Research Principle

The project is investigating whether multimodal computation can occur inside a shared recurrent transformer state.

The initial architecture should remain deliberately simple.

Do not introduce pretrained vision models, Hugging Face Transformers models, Q-Formers, Perceiver Resamplers, Mixture-of-Experts routing, or other large architectural dependencies unless explicitly requested.

The initial model should use:

* text token embeddings
* direct image patch embeddings
* a transformer prelude
* a weight-shared recurrent transformer core
* a transformer coda
* a language-model output head

The same recurrent core should operate over multimodal representations.

---

## Package Layout

All importable source code belongs under:

```text
src/multimodal_loop/
```

Use package imports such as:

```python
from multimodal_loop.model.config import ModelConfig
```

Do not create top-level Python packages named `model`, `train`, `data`, or `eval`.

Expected high-level structure:

```text
src/multimodal_loop/
├── model/
├── data/
├── train/
└── eval/
```

Tests belong in:

```text
tests/
```

Executable entry-point scripts belong in:

```text
scripts/
```

---

## Model Code Responsibilities

Keep modules focused.

### `model/config.py`

Canonical model configuration and validation.

Avoid passing large collections of unrelated constructor arguments throughout the codebase.

### `model/patch_embedding.py`

Image-to-token conversion.

The initial implementation should be simple and written directly in PyTorch.

Do not use a pretrained vision encoder.

### `model/attention.py`

Attention implementation and attention-mask construction.

Multimodal prefix masking must be independently testable.

### `model/transformer.py`

Reusable transformer blocks and stacks.

Keep the initial transformer implementation conventional unless the task explicitly concerns changing it.

### `model/recurrent_core.py`

Weight-shared recurrent-depth computation.

This module is expected to become a major research surface later.

### `model/model.py`

Defines the top-level `MultimodalLoopTransformer`.

This file should compose components rather than contain all implementation details.

---

## Implementation Style

Prefer:

* plain PyTorch
* explicit tensor shapes
* readable implementations
* small focused modules
* type annotations where useful
* deterministic behavior where practical
* clear assertions for shape assumptions
* simple solutions before optimized ones

Avoid unnecessary abstractions.

Do not add a dependency merely to save a few lines of code.

In particular, do not add the following unless explicitly required:

* `transformers`
* `accelerate`
* `lightning`
* `deepspeed`
* `xformers`
* `flash-attn`
* `einops`
* `torchvision`
* `wandb`

---

## Device Handling

The model must not assume CUDA.

Never write code such as:

```python
x = x.cuda()
```

Use device-independent PyTorch code.

Tensor placement should follow existing tensors, model parameters, or an explicitly supplied device.

The project is intended to run eventually on:

* CPU
* CUDA GPUs
* Kaggle GPUs
* Kaggle TPUs through PyTorch/XLA

TPU-specific infrastructure is not required during early Milestone 0 work unless explicitly requested.

---

## Dependencies

`pyproject.toml` is the authoritative package configuration.

The package should remain installable directly from GitHub using pip.

Do not unnecessarily alter or replace accelerator-specific PyTorch installations in hosted environments such as Kaggle.

---

## Testing

Correctness tests are required for core model behavior.

Run:

```bash
pytest
```

before considering a change complete.

Also run:

```bash
ruff check .
```

for linting.

If formatting changes are needed, use:

```bash
ruff format .
```

Important early tests include:

* configuration validation
* patch-embedding shapes
* attention-mask correctness
* recurrence parameter sharing
* variable recurrence depth
* forward-pass shapes
* backward-pass success
* text-only input
* multimodal input
* deterministic checkpoint save/load

Do not weaken or remove tests merely to make a change pass.

---

## Research Discipline

When modifying experimental behavior:

1. Preserve a simple baseline.
2. Change one major variable at a time when practical.
3. Make recurrence depth explicit and measurable.
4. Avoid silently changing computational budgets.
5. Keep dense and recurrent models comparable where relevant.
6. Expose experimental hyperparameters through configuration rather than hidden constants.

Do not optimize benchmark performance at the expense of understanding the mechanism.

---

## Recurrent Computation

The conceptual recurrent operation is:

```text
h_(r+1) = F_theta(h_r)
```

where the same parameters are reused at each recurrent step.

Do not accidentally instantiate separate transformer parameters for each recurrence unless implementing an explicit dense baseline.

Variable recurrent depth must remain possible at runtime.

---

## Multimodal Design

The initial model should treat images as patch tokens rather than representations from a pretrained vision model.

Conceptually:

```text
image
  ↓
patchify
  ↓
linear embedding
  ↓
visual tokens
  ┐
  ├── shared multimodal transformer state
  ┘
text tokens
```

For question-answer style training, image/question context may use bidirectional prefix attention while generated answer tokens remain causal.

Prevent target leakage.

Masking behavior must be covered by unit tests.

---

## Training

Early synthetic-vision training prioritizes correctness over speed.

Initially prefer:

* standard PyTorch
* standard cross-entropy
* straightforward optimizer usage
* single-process execution
* small synthetic tensors

Distributed training, mixed precision, compilation, kernel optimization, and TPU support should come only after the basic model and training path are validated.

---

## Documentation

Update documentation when a change materially alters:

* architecture
* repository structure
* training behavior
* configuration
* experiment methodology
* installation instructions

Do not duplicate large sections of `README.md` inside this file.

`README.md` is the human-facing research overview.

`AGENTS.md` is the operational guide for coding agents.

---

## Current Milestone

Milestone 0 is complete, including CPU correctness and Kaggle single-device CUDA
validation. TPU hardware validation is deferred.

Milestone 1 begins with deterministic single-object color questions, independently
tested rendering and labels, and disjoint training/validation/test layouts.
Scene metadata is for supervision and inspection, never model input. Establish
held-out visual grounding before investigating recurrence benefits in Milestone 2.

Preserve the infrastructure baseline:

* the package installs successfully
* imports work
* the model runs forward
* the model runs backward
* image and text inputs work
* recurrence depth is variable
* attention masks are tested
* checkpoints save and load
* tests pass reliably

Keep new research increments small and preserve these foundations as the data and training paths grow.

