# Multimodal Loop Transformer

An experimental multimodal foundation-model architecture built around **recurrent depth**.

The central goal of this project is to investigate whether a transformer can learn to perform useful iterative computation over a **shared multimodal latent state**, and whether increasing recurrent depth at inference time can improve multimodal reasoning without increasing the number of model parameters.

Rather than attaching a pretrained vision encoder to a pretrained language model, this project initially trains the entire system **from random initialization**.

Images and text are converted into tokens and processed by the same recurrent transformer architecture.

---

## Research Goal

Modern multimodal language models usually follow a structure similar to:

```text
Image
  ↓
Pretrained Vision Encoder
  ↓
Projection / Resampler
  ↓
Language Model
  ↓
Text
```

This project explores a different design:

```text
        Image                         Text
          │                            │
          ▼                            ▼
     Patch Embedding              Token Embedding
          │                            │
          └─────────────┬──────────────┘
                        ▼
                 Multimodal State
                        │
                        ▼
                    Prelude
                        │
                        ▼
                ┌─────────────┐
                │             │
                │  Recurrent  │
          ┌────►│ Transformer │────┐
          │     │    Core     │    │
          │     │             │    │
          │     └─────────────┘    │
          │                        │
          └────── repeat R times ◄─┘
                        │
                        ▼
                      Coda
                        │
                        ▼
                     Output
```

The recurrent core uses the **same parameters at every recurrent step**.

Conceptually,

$$
h_{r+1} = F_\theta(h_r)
$$

where \(h_r\) contains both linguistic and visual representations.

The hope is that successive recurrent steps can perform progressively deeper multimodal computation:

```text
Loop 1 → identify relevant visual features
Loop 2 → establish relationships
Loop 3 → combine visual and linguistic information
Loop 4 → perform reasoning
Loop 5 → refine the answer
```

These stages are not explicitly programmed into the model. They are behaviors we want to determine whether the model can learn.

---

# Primary Research Questions

The initial project is designed around several questions.

### 1. Can multimodal representations emerge from random initialization?

The first model will not use a pretrained vision encoder.

Images will be patchified and linearly embedded:

$$
\text{image} \rightarrow \text{patches} \rightarrow \mathbb{R}^{d_\text{model}}
$$

These visual tokens will enter the same model as language tokens.

We want to determine whether useful visual representations can emerge purely through end-to-end multimodal training.

---

### 2. Does recurrent depth improve multimodal reasoning?

The primary hypothesis is:

$$
\text{performance}(R + 1) > \text{performance}(R)
$$

for tasks requiring additional reasoning.

A successful model should ideally require little recurrent computation for easy tasks and benefit from additional computation for harder ones.

---

### 3. Does useful inference-time depth generalization occur?

The model will be trained with randomized recurrent depth.

At evaluation time, it will be tested with recurrence depths both inside and outside the training distribution.

For example:

```text
Training:
R ≈ 4–8

Evaluation:
R = 1
R = 2
R = 4
R = 8
R = 12
R = 16
R = 24
R = 32
```

An especially interesting result would be continued improvement at recurrence depths larger than those typically encountered during training.

---

### 4. Does task complexity correlate with useful recurrent depth?

Synthetic multimodal tasks will be constructed with explicitly controllable reasoning depth.

For example:

```text
1-hop:
"What color is the square?"

2-hop:
"What shape is left of the blue circle?"

4-hop:
"What color is the object below the shape
left of the triangle nearest the red square?"
```

This allows us to study the relationship:

$$
\text{reasoning complexity}
\quad\leftrightarrow\quad
\text{useful recurrent depth}
$$

---

# Design Principles

## Keep the first model simple

The initial architecture intentionally avoids:

* pretrained vision encoders
* Q-Formers
* Perceiver Resamplers
* Mixture-of-Experts routing
* image reconstruction decoders
* complicated multimodal adapters
* large-scale instruction tuning
* learned halting mechanisms

These may be explored later.

The first objective is to isolate **multimodal recurrent computation**.

---

## Change one major variable at a time

The long-term project may investigate:

* new recurrent architectures
* alternative attention mechanisms
* new optimizers
* new training algorithms
* learned computation allocation
* modality-specific recurrence
* alternative objectives
* non-transformer sequence architectures

However, early experiments should retain known-good components wherever possible.

A failed experiment should tell us *what failed*.

---

## Small models before large models

Approximate experimental progression:

```text
20–50M parameters
    ↓
debugging and synthetic experiments

50–150M
    ↓
primary architecture experiments

300–500M
    ↓
meaningful scaling experiments

1–3B
    ↓
validation of scaling behavior

7B+
    ↓
only if earlier experiments justify it
```

Large-scale training should not begin until the recurrent mechanism demonstrates measurable value at small scale.

---

# Initial Architecture

A simplified model interface may look like:

```python
class MultimodalLoopTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_embed = TokenEmbedding(config)
        self.image_embed = PatchEmbedding(config)

        self.prelude = TransformerStack(config)
        self.core = RecurrentTransformerCore(config)
        self.coda = TransformerStack(config)

        self.lm_head = LMHead(config)

    def forward(
        self,
        input_ids,
        images=None,
        recurrence_depth=None,
        attention_mask=None,
    ):
        text = self.text_embed(input_ids)

        if images is not None:
            vision = self.image_embed(images)
            x = combine(vision, text)
        else:
            x = text

        x = self.prelude(x, attention_mask)

        for _ in range(recurrence_depth):
            x = self.core(x, attention_mask)

        x = self.coda(x, attention_mask)

        return self.lm_head(x)
```

This is intentionally only a conceptual skeleton.

The recurrent implementation, positional representations, masking scheme, initialization, and training dynamics are all expected to evolve.

---

# Multimodal Attention

For visual question answering, the initial model will use a **multimodal prefix attention mask**.

Given:

```text
[IMAGE TOKENS] [QUESTION TOKENS] [ANSWER TOKENS]
```

the image and question form a bidirectional context.

```text
IMAGE TOKENS
     ↕
QUESTION TOKENS
```

The answer remains autoregressive.

Conceptually:

```text
IMAGE    → IMAGE + QUESTION
QUESTION → IMAGE + QUESTION

ANSWER 1 → IMAGE + QUESTION
ANSWER 2 → IMAGE + QUESTION + ANSWER 1
ANSWER 3 → IMAGE + QUESTION + ANSWER 1 + ANSWER 2
```

Image and question tokens must never attend to future answer tokens.

The purpose of this masking scheme is to allow visual representations themselves to change during recurrent computation.

---

# Repository Structure

Importable code lives under `src/multimodal_loop/`. Configuration and patch
embedding are implemented; the remaining model, data, training, and evaluation
modules are scaffolding for subsequent increments.

```text
multimodal-loop/
│
├── README.md
│
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── debug.yaml
│   ├── tiny.yaml
│   └── base.yaml
│
├── src/
│   └── multimodal_loop/
│       ├── __init__.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── embeddings.py
│       │   ├── patch_embedding.py
│       │   ├── attention.py
│       │   ├── transformer.py
│       │   ├── recurrent_core.py
│       │   └── model.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── synthetic_shapes.py
│       │   ├── text.py
│       │   ├── multimodal.py
│       │   └── collator.py
│       ├── train/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── recurrence.py
│       │   ├── losses.py
│       │   └── schedules.py
│       └── eval/
│           ├── __init__.py
│           ├── synthetic.py
│           ├── recurrence_sweep.py
│           ├── stability.py
│           └── diagnostics.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate_synthetic_data.py
│
└── tests/
    ├── test_config.py
    ├── test_attention_mask.py
    ├── test_patch_embedding.py
    ├── test_recurrence.py
    └── test_model.py
```

---

# Module Responsibilities

The paths below are relative to `src/multimodal_loop/`.

## `model/config.py`

Defines the immutable `ModelConfig` dataclass, validates architecture settings,
and derives attention-head width, patch count, and flattened patch size.

---

## `model/model.py`

Defines the complete `MultimodalLoopTransformer`.

Responsibilities include:

* connecting embeddings to the model
* combining visual and text tokens
* running the prelude
* invoking recurrent computation
* running the coda
* producing model outputs

This file should describe the architecture at the highest level rather than contain detailed implementations.

---

## `model/patch_embedding.py`

Converts images directly into model tokens.

Initially:

```text
RGB Image
    ↓
Fixed-size patches
    ↓
Flatten / project
    ↓
d_model-dimensional visual tokens
```

No pretrained image representation is assumed.

---

## `model/attention.py`

Contains:

* self-attention implementation
* causal masks
* multimodal prefix masks
* optional future attention experiments

Attention behavior should be independently testable.

---

## `model/transformer.py`

Contains the basic transformer block used by the model.

Initially this should remain relatively conventional so that recurrence can be studied independently.

---

## `model/recurrent_core.py`

Contains the repeated transformation:

$$
h_{r+1}=F_\theta(h_r)
$$

The same parameters are reused across recurrent steps.

This module will eventually become one of the primary areas of architectural research.

---

## `train/recurrence.py`

Controls the training-time recurrence policy.

Examples may eventually include:

* fixed recurrence
* uniform random recurrence
* log-normal recurrence
* curriculum schedules
* truncated backpropagation
* stochastic truncation
* adaptive recurrence
* learned halting

Initially, a Huginn-like randomized recurrence strategy will serve as the baseline.

---

# Initial Training Objective

The first experiment will deliberately use a simple objective:

$$
\mathcal{L}=\mathcal{L}_{CE}
$$

Cross-entropy is calculated over target language tokens.

No auxiliary visual objective is required initially.

This tests whether meaningful visual representations can emerge from language supervision alone.

Future experiments may introduce:

$$
\mathcal L =
\mathcal L_{\text{text}}
+
\lambda_v\mathcal L_{\text{vision}}
+
\lambda_a\mathcal L_{\text{alignment}}
+
\lambda_r\mathcal L_{\text{recurrent}}
$$

but auxiliary losses should be treated as experimental variables rather than assumptions.

---

# Training Data

Training will progress through several stages.

## Stage 0 — Synthetic multimodal reasoning

Procedurally generated images will contain simple objects with controlled attributes such as:

* color
* shape
* size
* position
* count
* spatial relationships

Corresponding questions can have precisely controlled reasoning depth.

Example:

```text
Image:
red circle
blue square
green triangle

Question:
"What color is the object left of the green triangle?"

Answer:
"blue"
```

Advantages:

* effectively unlimited data
* perfect ground truth
* controlled task complexity
* controlled distribution shifts
* no contamination
* no licensing restrictions
* exact measurement of reasoning depth

---

## Stage 1 — Image-caption pretraining

Candidate datasets include open multimodal caption corpora such as PixMo.

The purpose is to teach richer real-world visual semantics.

---

## Stage 2 — Mixed multimodal training

Training batches may contain a mixture of:

```text
Text-only language modeling

Image → caption

Image + question → answer

Image + multi-step reasoning → answer
```

The same recurrent core handles every case.

---

## Stage 3 — Large multimodal mixtures

Once the architecture has demonstrated useful behavior at smaller scale, larger curated multimodal datasets may be introduced.

Large-scale training should be treated as validation of the architecture rather than the mechanism by which we discover whether it works.

---

# Baselines

Every major recurrent experiment should have matched controls.

At minimum:

| Model            | Recurrent | Multimodal |
| ---------------- | --------: | ---------: |
| Language Dense   |        No |         No |
| Language Loop    |       Yes |         No |
| Multimodal Dense |        No |        Yes |
| Multimodal Loop  |       Yes |        Yes |

The most important comparison is:

```text
Multimodal Dense
       vs.
Multimodal Loop
```

---

# Comparison Regimes

## Iso-parameter

Models have approximately equal trainable parameter counts.

This asks:

> Does recurrent computation use parameters more effectively?

---

## Iso-FLOP

Models receive approximately equal computational budgets.

This asks:

> Does recurrent computation provide an advantage when compute is controlled?

Both comparisons are necessary.

---

# Recurrence Evaluation

Each trained recurrent checkpoint should be evaluated over a sweep of recurrent depths.

For example:

```text
R = 1
R = 2
R = 3
R = 4
R = 6
R = 8
R = 12
R = 16
R = 24
R = 32
```

Performance should be plotted against recurrence depth.

The evaluation suite should also separate tasks by reasoning complexity.

Desired behavior might resemble:

```text
             Recurrence Depth

Task         2     4     8     16
----------------------------------
Color       94    95    95     94
Count       75    83    84     83
2-hop       61    73    78     78
4-hop       35    51    69     74
6-hop       21    31    48     63
```

The exact numbers are illustrative.

The important property is that deeper reasoning tasks should benefit more strongly from additional recurrent computation.

---

# Recurrent-State Diagnostics

Accuracy alone is insufficient.

We also want to understand what repeated computation does internally.

Potential metrics include:

### State change

$$
\Delta_r =
\frac{\|h_{r+1}-h_r\|}
{\|h_r\|}
$$

and modality-specific variants:

$$
\Delta_r^\text{vision}
$$

$$
\Delta_r^\text{text}
$$

### Additional diagnostics

* activation norms by recurrence
* gradient norms by recurrence
* attention from text to image
* attention from image to text
* output entropy by recurrence
* KL divergence between successive output distributions
* cosine similarity between successive hidden states
* modality-specific representation drift
* recurrence-dependent confidence

These measurements should help distinguish useful iterative refinement from unstable repeated computation.

---

# Stability

Recurrent transformers can exhibit unstable dynamics as recurrent depth increases.

Stability should therefore be treated as a first-class research problem.

Every experiment should monitor:

* exploding activations
* vanishing representations
* gradient growth
* state collapse
* modality collapse
* oscillatory behavior
* degradation at large recurrence depth

Methods inspired by recurrent-depth architectures such as Huginn and stabilization techniques such as those explored by Parcae are important reference points.

---

# Initial Model Scale

A reasonable first serious model might use approximately:

```text
d_model:            512–768
attention heads:    8–12

prelude blocks:     2
recurrent blocks:   2
coda blocks:        2

nominal recurrence: 4–8

image resolution:   112×112
patch size:         16×16
visual tokens:      49

parameter target:   50–150M
```

Smaller 20–50M parameter configurations should be used during debugging.

---

# Milestones

## Milestone 0 — Infrastructure

* model runs forward and backward
* image and text batches work
* recurrent depth can change dynamically
* masking tests pass
* checkpoints save and load
* training can resume deterministically

---

## Milestone 1 — Synthetic vision

Train a small model from random initialization to answer basic visual questions.

Success criterion:

> The model learns meaningful visual grounding without a pretrained vision encoder.

---

## Milestone 2 — Multimodal recurrence

Introduce multi-hop synthetic questions.

Success criterion:

> Additional recurrent steps improve performance on tasks requiring additional reasoning.

This is the project's first major scientific milestone.

---

## Milestone 3 — Depth generalization

Train using a restricted distribution of recurrent depths and evaluate beyond it.

Success criterion:

> Inference-time recurrence greater than typical training recurrence produces useful computation rather than immediate degradation.

---

## Milestone 4 — Dense comparison

Train matched dense and recurrent multimodal models.

Success criterion:

> Establish whether recurrent depth provides a measurable parameter-efficiency, compute-efficiency, reasoning, or generalization advantage.

---

## Milestone 5 — Real images

Introduce real image-caption and VQA data.

Success criterion:

> The architectural behavior observed synthetically survives contact with real multimodal data.

---

## Milestone 6 — New training algorithms

Once the baseline is well understood, begin replacing the training machinery.

Potential research areas include:

* recurrence-aware optimization
* dynamically sampled computation
* recurrent-state regularization
* adaptive truncation
* learned halting
* modality-dependent computation budgets
* stability-aware objectives
* curriculum over reasoning depth
* alternative credit-assignment methods

---

## Milestone 7 — Scaling

Only after the architecture has survived controlled experimentation should it be scaled into the hundreds of millions or billions of parameters.

---

# What Would Count as Success?

The project does **not** initially need to outperform state-of-the-art multimodal models.

The first architecture would be considered scientifically successful if:

1. Visual grounding emerges from random initialization.
2. The same recurrent core handles text and visual representations.
3. Increasing recurrence improves some multimodal reasoning tasks.
4. Harder tasks benefit from greater recurrent depth than easier tasks.
5. Useful computation occurs beyond commonly seen training depths.
6. Recurrent states remain stable over substantial unrolling.
7. The recurrent model behaves measurably differently from a matched dense model.

If these properties appear consistently, scaling becomes justified.

---

# Long-Term Direction

The long-term goal is not merely to build:

```text
vision encoder + looped LLM
```

but to investigate a model in which **multimodal cognition itself is recurrent**.

Potential future architectures could allow computation to be allocated dynamically:

```text
Visual token A   ███████████████  7 steps
Visual token B   █████████        5 steps
Text token A     █████            3 steps
Text token B     █████████████    6 steps
```

Different modalities, tokens, or latent states could receive different amounts of computation depending on task difficulty.

Ultimately, the project asks whether a multimodal foundation model can learn not only **what representations to construct**, but also **how long to think about them**.

---

# Influences

The project is particularly inspired by research into:

* Universal Transformers
* Huginn / recurrent-depth language models
* Parcae and recurrent-depth stabilization
* LLaVA-style multimodal token integration
* nanoVLM
* Molmo / PixMo
* parameter-sharing multimodal transformers

These projects provide useful reference implementations and experimental precedents, but the objective here is to build a native multimodal recurrent architecture rather than attach multimodal adapters to an existing pretrained model.

---

# Status

**Phase:** Milestone 0 — infrastructure and correctness, in progress.

The first increment implements validated model configuration and direct image
patch embeddings, with tests for patch ordering, projection, input validation,
and gradient flow. The complete transformer, attention masks, recurrent core,
training, and checkpoint support remain to be implemented.

## Local development

Python 3.11 or newer is required. For a local CPU development environment, run
these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.7,<3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the authoritative dependency configuration. PyTorch is an
optional dependency so that hosted environments such as Kaggle can retain their
accelerator-specific installation. If compatible PyTorch is already installed,
use that environment and run only the editable-install command above. The
`requirements.txt` alternative includes the `torch` extra for local environments
where pip should provide PyTorch.

Run the checks in the activated environment:

```bash
pytest
ruff check .
ruff format --check .
```

If formatting needs to change, run `ruff format .`.

## Configuration and patch embedding

`ModelConfig()` uses deliberately tiny defaults for CPU correctness work:

| Settings | Defaults |
| --- | --- |
| `vocab_size`, `max_seq_len` | 256, 128 |
| `d_model`, `n_heads`, `d_ff` | 64, 4, 256 |
| `n_prelude_layers`, `n_recurrent_layers`, `n_coda_layers` | 1, 1, 1 |
| `recurrence_depth` | 2 |
| `image_size`, `patch_size`, `num_channels` | 32, 8, 3 |
| `dropout`, `layer_norm_eps` | 0.0, 1e-5 |

The larger dimensions in Initial Model Scale describe later experiments, not the
current defaults. `max_seq_len` is the eventual combined visual/text sequence
limit. `recurrence_depth` specifies the default number of applications of the
shared recurrent stack; runtime overrides will be supported by the full model.

```python
import torch

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.patch_embedding import PatchEmbedding

config = ModelConfig()
image_embed = PatchEmbedding(config)
images = torch.randn(2, config.num_channels, config.image_size, config.image_size)
tokens = image_embed(images)

assert tokens.shape == (2, 16, 64)
tokens.square().mean().backward()
```

Each configuration accepts one fixed square image size, divisible by its square
patch size. Input tensors must be floating point and shaped `[batch, channels,
height, width]`. Patches are ordered left to right, then top to bottom; values
within a patch are flattened in channel, row, column order. One shared linear
projection with bias maps every patch to `d_model` features. Positional and
modality embeddings will be added separately.

Inputs and module parameters follow normal PyTorch device and dtype conventions;
patch embedding performs no implicit transfers or casts. Configurations are
immutable and can be serialized with `dataclasses.asdict(config)` and rebuilt
with `ModelConfig(**values)`.

Immediate objective:

> Build a small end-to-end multimodal loop transformer and determine whether recurrent depth improves controlled synthetic visual reasoning.

Scale comes later.

First, prove the loop matters.
# multi-modal-loop-llm
Multi-Modal Loop LLM experiment
