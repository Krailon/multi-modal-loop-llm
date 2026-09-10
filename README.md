# Multimodal Loop Transformer

An experimental multimodal foundation-model architecture built around **recurrent depth**.

The primary goal is to progressively build and train a **usable multimodal model**
that supports increasingly complex long-term awareness experiments. Progress will
be measured through concrete language, perception, reasoning, and eventually
memory capabilities. Broad general capability is an ambition, not a promised
level of intelligence.

The initial research question is whether iterative computation over a **shared
multimodal latent state** offers practical advantages, disadvantages, or tradeoffs
compared with alternatives. A recurrence advantage is not required for the model
to be useful or for capability development to continue.

Rather than attaching a pretrained vision encoder to a pretrained language model, this project initially trains the entire system **from random initialization**.

Images and text are converted into tokens and processed by the same recurrent transformer architecture.

---

## Functional Goal and Initial Research Direction

Functional progress and architectural findings are assessed separately. Build
reliable capabilities and a controllable experimental platform, while testing
which architectural choices help those capabilities under matched conditions.
Positive, negative, and neutral recurrence findings are all informative.

The current recurrence repeats computation **within one forward pass**. It does
not yet maintain persistent memory across interactions. Long-term awareness
experiments will require separately designed and evaluated capabilities such as
memory, temporal continuity, and adaptation; recurrent depth alone does not
establish those capabilities.

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

# Research Questions

The initial architectural question is whether shared multimodal recurrence is
useful in practice, including its costs and failure modes. The questions below
are starting hypotheses and investigation directions, not a checklist of positive
results required for project success. Additional questions can be introduced as
the functional model and experimental needs develop.

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

One hypothesis to test is:

$$
\text{performance}(R + 1) > \text{performance}(R)
$$

for tasks requiring additional reasoning.

An interesting outcome would be little recurrent computation for easy tasks and
benefits from additional computation for harder ones. No improvement or degradation
would also be valid findings; neither prevents pursuing functional capabilities.

---

### 3. Does useful inference-time depth generalization occur?

A future depth-generalization experiment can train with randomized recurrent depth.
The current capability baseline trains at a fixed depth.

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

The first objective is a functional, understandable multimodal baseline with
which to investigate **multimodal recurrent computation**.

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

Large-scale training should wait for reliable small-scale capabilities, understood
training behavior, and an explicit resource budget. A measured recurrence advantage
is not a prerequisite for continued capability development.

---

# Initial Architecture

The implemented model composes the components as follows:

```python
class MultimodalLoopTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = MultimodalEmbedding(config)

        self.prelude = TransformerStack(config, config.n_prelude_layers)
        self.core = RecurrentTransformerCore(config)
        self.coda = TransformerStack(config, config.n_coda_layers)

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        images=None,
        recurrence_depth=None,
        attention_mask=None,
    ):
        x = self.embeddings(input_ids, images)
        if attention_mask is None:
            attention_mask = build_causal_mask(x.shape[1], device=x.device)

        x = self.prelude(x, attention_mask)

        x = self.core(x, attention_mask, recurrence_depth=recurrence_depth)

        x = self.coda(x, attention_mask)

        return self.lm_head(self.final_norm(x))
```

This is a condensed view of `MultimodalLoopTransformer` in
`src/multimodal_loop/model/model.py`. The final LayerNorm operates independently
at each sequence position. The vocabulary projection is bias-free and has
independent weights from the text embeddings. Both use standard PyTorch
initialization. The model returns raw logits for all image and text positions.

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

Image and question tokens must never attend to any answer tokens.

The diagram describes answer prediction. Actual attention masks index **input
tokens** and include the diagonal: each input can attend to itself, and its
output predicts the next token. Loss computation is external to the model:
callers shift targets by one position. For question answering, the final
question input predicts the first answer token, with loss applied to answer
targets.

`build_causal_mask(seq_len, *, device=None)` and
`build_prefix_mask(seq_len, prefix_len, *, device=None)` are implemented in
`multimodal_loop.model.attention`. They return boolean `[seq_len, seq_len]`
tensors with query rows and key columns. `True` means attention is allowed,
matching PyTorch's scaled dot-product attention convention. The prefix length
counts all image/question input tokens. A zero-length prefix gives causal
attention; a full-length prefix permits all attention. Each mask uses one
prefix length shared across batch items, without padding handling.

`MultimodalLoopTransformer` builds one causal mask when no mask is supplied,
including for image inputs. To enable bidirectional image/question context,
callers supply a prefix mask whose boundary excludes all answer inputs. The
same mask is passed through the prelude, every recurrent step, and the coda.
The model validates the mask's shape, dtype, device, and nonempty query rows;
choosing the correct context boundary remains the caller's responsibility.

The purpose of this masking scheme is to allow visual representations themselves to change during recurrent computation.

## Self-attention

`SelfAttention(config)` in `multimodal_loop.model.attention` implements
conventional multi-head self-attention. Its `forward(x, attention_mask=None)`
method accepts floating-point `[batch, seq_len, d_model]` inputs and returns
transformed representations with the same shape. Omitting the mask uses causal
attention; supplying a prefix mask enables the multimodal behavior above.

One learned linear projection produces queries, keys, and values for all heads.
PyTorch's `scaled_dot_product_attention` computes attention, and a second linear
projection combines the heads. Both projections use bias and standard PyTorch
initialization. `config.dropout` controls attention-probability dropout during
training; calling `.eval()` disables dropout.

Explicit masks must be boolean `[seq_len, seq_len]` tensors on the input device,
with at least one allowed key in every query row. Inputs and parameters follow
normal PyTorch device and dtype conventions, without implicit transfers or
casts. `TransformerBlock` provides residual connections and normalization;
`MultimodalEmbedding` adds positional embeddings before the transformer stacks.

---

# Repository Structure

Importable code lives under `src/multimodal_loop/`. Configuration, patch and
multimodal embeddings, attention-mask helpers, self-attention, transformer
blocks/stacks, the recurrent core, and the complete language-model composition
are implemented, along with shifted language loss, a minimal tensor-batch
training loop, and single-device checkpoint/resume support. Data, evaluation, and
training-policy components remain scaffolding for subsequent increments.

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
│       │   ├── runtime.py
│       │   ├── checkpoint.py
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
    ├── test_attention.py
    ├── test_patch_embedding.py
    ├── test_embeddings.py
    ├── test_transformer.py
    ├── test_recurrence.py
    ├── test_model.py
    ├── test_losses.py
    ├── test_trainer.py
    └── test_checkpoint.py
```

---

# Module Responsibilities

The paths below are relative to `src/multimodal_loop/`.

## `model/config.py`

Defines the immutable `ModelConfig` dataclass, validates architecture settings,
and derives attention-head width, patch count, and flattened patch size.

---

## `model/model.py`

Defines `MultimodalLoopTransformer(config)` with
`forward(input_ids, images=None, recurrence_depth=None, attention_mask=None)`.

Responsibilities include:

* invoking shared multimodal embedding assembly
* running the prelude
* invoking recurrent computation
* running the coda
* applying final normalization and projecting to vocabulary logits

It returns `[batch, text_length, vocab_size]` for text only, or
`[batch, num_patches + text_length, vocab_size]` with images. Logits preserve the
image-first sequence layout. Loss computation, target shifting, and selecting
supervised positions belong to callers. This module composes existing
components and delegates input, mask, and recurrence-depth validation to them.

---

## `model/embeddings.py`

`MultimodalEmbedding` combines optional image patches and required text tokens
into one sequence. It adds learned 1D positional embeddings and two learned
modality embeddings, then returns the state consumed by the transformer prelude.
Sequence construction happens once, before recurrent computation.

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

Defines `TransformerBlock(config)`, which adds normalization, a feed-forward
network, and residual connections around `SelfAttention`. Its
`forward(x, attention_mask=None)` method preserves `[batch, seq_len, d_model]`
and leaves the input tensor unchanged. The attention layer supplies the causal
default and validates explicit masks.

Each branch uses its own trainable LayerNorm over the feature dimension, before
computing its update:

```text
h = x + dropout_attention(SelfAttention(LayerNorm_attention(x), attention_mask))
y = h + dropout_feedforward(FFN(LayerNorm_feedforward(h)))
```

The feed-forward network is `Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)`.
Both projections use bias and standard PyTorch initialization. The two LayerNorm
modules use `config.layer_norm_eps` and independent affine parameters.
`config.dropout` controls dropout on each branch output before residual addition,
as well as attention-probability dropout inside `SelfAttention`. Calling `.eval()`
disables all dropout. The block returns `y` directly.

`TransformerStack(config, num_layers)` constructs that many independent blocks
and applies them in order with `forward(x, attention_mask=None)`. It forwards
the same mask to every block and returns the final `[batch, seq_len, d_model]`
representation directly. There is no additional normalization or residual
connection around the stack.

The explicit layer count must be a nonnegative integer, excluding booleans.
A zero-layer stack is a parameter-free identity, supporting prelude or coda
configurations with no blocks. It still validates input shapes, dtypes, and
explicit masks. Stacks leave the input unchanged and use normal PyTorch device
and dtype handling.

---

## `model/recurrent_core.py`

`RecurrentTransformerCore(config)` owns one `TransformerStack` with
`config.n_recurrent_layers` independent blocks. Its
`forward(x, attention_mask=None, *, recurrence_depth=None)` method applies the
whole stack repeatedly:

$$
h_{r+1}=F_\theta(h_r)
$$

The same stack parameters are reused at every recurrent step. The core owns
the repetition loop, so the top-level model calls it once with the requested
depth. An omitted depth uses `config.recurrence_depth`; overrides must be
positive integers, excluding booleans, and may exceed that default.

The same mask is passed on every repetition, and the complete autograd graph
is preserved through all steps. Training dropout uses normal random draws on
each repetition, while `.eval()` disables it. The core returns only the final
state, preserving `[batch, seq_len, d_model]` and leaving the input unchanged.
Changing runtime depth changes the computation count without changing parameter
identities, parameter count, state-dictionary structure, or configuration.

The model's total block applications per forward pass are:

```text
n_prelude_layers + recurrence_depth * n_recurrent_layers + n_coda_layers
```

For example, two blocks in the recurrent stack repeated five times execute ten
block applications using the parameters of just those two blocks. Depth sampling
policies remain a separate training concern.

```python
import torch

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.recurrent_core import RecurrentTransformerCore

config = ModelConfig(n_recurrent_layers=2)
core = RecurrentTransformerCore(config)
hidden = torch.randn(2, 6, config.d_model, requires_grad=True)
output = core(hidden, recurrence_depth=5)

assert output.shape == hidden.shape
output.square().mean().backward()
```

This module will eventually become one of the primary areas of architectural research.

---

## `train/losses.py` and `train/trainer.py`

`shifted_cross_entropy` aligns image-first vocabulary logits with next-token
text targets and averages over selected targets across the batch. An optional
boolean mask selects target token positions before shifting.

`train_on_batch` repeats standard optimizer steps on one fixed tensor batch.
It constructs compatible attention and supervision masks from a shared question
length, clears parameter gradients on every step, and retains the caller's
optimizer state across calls. This is the Milestone 0 training smoke path.

---

## `train/checkpoint.py`

`save_checkpoint` writes configuration, model and AdamW state, the fixed tensor
batch, completed steps, training settings, and global RNG states to a single
versioned `.pt` file. `load_checkpoint(path, device="cpu")` reconstructs a model and optimizer
on the requested matching backend
and returns them in `LoadedCheckpoint` with the saved batch and progress.
RNG restoration happens after object construction so initialization cannot
consume the resumed random sequence.

---

## `train/recurrence.py`

Reserved for future training-time recurrence policies.

Examples may eventually include:

* fixed recurrence
* uniform random recurrence
* log-normal recurrence
* curriculum schedules
* truncated backpropagation
* stochastic truncation
* adaptive recurrence
* learned halting

Milestone 0 uses a fixed, explicit depth for each training call. A Huginn-like
randomized recurrence strategy remains the intended baseline for later
experiments, after the basic training and checkpoint path is validated.

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

The roadmap combines capability development and research investigations. Later
research milestones can produce positive, negative, or neutral findings, and their
ordering can evolve with capability needs. They do not require an architectural
advantage before work on a useful model can continue.

## Milestone 0 — Infrastructure

**Status: Complete.** CPU correctness and deterministic resume are verified,
and single-device CUDA validation passed on Kaggle. TPU hardware validation is
deferred and does not block milestone completion.

* model runs forward and backward
* image and text batches work
* recurrent depth can change dynamically
* masking tests pass
* checkpoints save and load
* training can resume deterministically

---

## Milestone 1 — Synthetic vision

**Complete:** single-object color grounding on held-out layouts.
[Results and close-out review](docs/milestones/milestone1.md).

Train a small model from random initialization to answer basic visual questions.

Success criterion:

> The model learns meaningful visual grounding without a pretrained vision encoder.

---

## Milestone 2 — Question-dependent visual reasoning

Build on single-object color grounding with varied questions about multi-object
scenes. Start with deterministic scenes, unambiguous one-hop relational questions,
and independently verified labels, then establish learning at a fixed recurrence
depth on held-out examples.

Success criterion:

> The model answers varied questions about multi-object scenes on held-out examples,
> with controls demonstrating that both the image and the question affect its
> answers appropriately.

Use image controls and same-image/different-question examples that require
different answers. This tests question dependence as well as image dependence.
The deterministic corpus and fixed-depth training/control infrastructure are
implemented. The [first baseline protocol](docs/milestones/milestone2.md) fixes
the training budget and validation acceptance gates. The first Kaggle baseline
completed but missed four gates; the completed diagnosis motivates a
[direct shape-grounding sanity experiment](docs/milestones/milestone2_shape_grounding.md).

Recurrence comparisons can follow as research. Demonstrating an advantage from
additional recurrent steps is not required to complete this capability milestone.

---

## Milestone 3 — Depth generalization

Train using a restricted distribution of recurrent depths and evaluate beyond it.

Success criterion:

> Characterize accuracy, stability, and computational cost beyond the training
> depth distribution, including improvements, neutral effects, and degradation.

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

Once the baseline is well understood, evaluate changes to the training machinery
when a concrete capability need or research question motivates them.

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

Functional success means a progressively more usable model and experimental
platform: reliable visual grounding, question-dependent reasoning, broader
language capability, and eventually explicitly tested memory and behavior across
extended interactions. Define each increment through observable tasks, held-out
evaluation, and controls. The project does not initially need to outperform
state-of-the-art models or meet an undefined threshold of “average intelligence.”

Research success means credible answers to well-specified questions. Tests of
recurrence should establish where it helps, has no meaningful effect, or hurts,
with attention to parameter count, computational cost, stability, and generalization.
A useful model remains a success even if recurrence offers no measured advantage.

Scaling decisions should follow demonstrated capabilities, training reliability,
and available resources. No particular recurrence hypothesis must be confirmed
before continuing model development.

---

# Long-Term Direction

The long-term functional goal is a capable multimodal model for increasingly
complex long-term awareness experiments. Potential future research questions
include:

* How should a model retain, retrieve, update, and forget information across interactions?
* How can temporal continuity and adaptation be evaluated without confusing them
  with retrieval or memorization of a fixed dataset?
* How should computation be allocated across modalities, tokens, or recurrent steps?
* Which architectural and training choices improve useful behavior, reliability,
  or efficiency under controlled comparisons?

This list is open-ended. Questions and priorities can evolve as the model becomes
more capable; these are possibilities, not implemented features or committed designs.
Persistent memory and cross-interaction behavior will need explicit mechanisms
and tests beyond the current recurrent-depth computation.

Shared multimodal recurrence remains the initial architectural investigation.
Future experiments may explore dynamic computation allocation or other designs,
while retaining simple baselines and changing one major variable at a time.

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

**Phase:** Milestone 2 in progress — the fixed-depth relational baseline has been
evaluated and diagnosed on Kaggle; direct grounding also showed a transfer gap.
The next experiment expands training geometry diversity.

**Milestone 0:** Infrastructure and correctness, officially complete.

**Milestone 1:** Complete for single-object color grounding on held-out layouts.
See [results, evidence, and close-out review](docs/milestones/milestone1.md).

Validated model configuration, direct image patch embeddings, shared image/text
sequence construction with learned positional and modality embeddings,
causal/prefix attention-mask helpers, multi-head self-attention, transformer
blocks/stacks, the recurrent core, and the complete model are implemented. Tests
cover patch ordering, projection, exact embedding sums, sequence capacity, input validation,
attention-math and PyTorch block-reference agreement, residual identity,
causal/prefix isolation, dropout behavior, runtime depth, parameter sharing,
and full gradient accumulation across repetitions. End-to-end tests use shifted
language-token cross-entropy for text-only and multimodal inputs, including
gradients to image pixels. Leakage tests verify protected states after every
recurrence and final output logits, while positive controls verify that context
can influence answer predictions. Shifted cross-entropy, answer-target masking,
and a small AdamW training script are implemented and tested on fixed synthetic
batches, including loss reduction and optimizer-state continuity across calls.
CPU checkpoint save/load and deterministic training resume are implemented.
Single-device CUDA/TPU training and same-backend resume paths are also implemented;
single-device CUDA validation passed in a Kaggle 2×T4 environment. The
user-confirmed hardware test result was `4 passed in 69.91s (0:01:09)`.
TPU hardware validation is deferred and does not block Milestone 0 completion.
CPU tests compare uninterrupted and resumed training with dropout enabled, including
exact loss, parameter, optimizer-state, and RNG equality across process restarts.
The package also builds and installs as a wheel, with imports, training, and a
checkpoint round trip verified outside the source checkout.

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
current defaults. `max_seq_len` limits the combined visual/text sequence length
in `MultimodalEmbedding`. `recurrence_depth` specifies the default number of
applications of the shared recurrent stack; `RecurrentTransformerCore` accepts
runtime overrides.

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
projection with bias maps every patch to `d_model` features. `MultimodalEmbedding`
adds positional and modality embeddings to these projected patches and to text.

Inputs and module parameters follow normal PyTorch device and dtype conventions;
patch embedding performs no implicit transfers or casts. Configurations are
immutable and can be serialized with `dataclasses.asdict(config)` and rebuilt
with `ModelConfig(**values)`.

## Shared image/text embeddings

`MultimodalEmbedding(config)` accepts `input_ids` shaped `[B, T]` and optional
`images` shaped `[B, C, H, W]`, with one image per batch item. Text IDs must be
`torch.int32` or `torch.int64`, with `T > 0` and values in `[0, vocab_size)`.
Every vocabulary ID, including zero, is an ordinary trainable token; there is
no padding behavior or automatic insertion of special tokens.

The output layout is `[image patches][text tokens]`, shaped `[B, P + T, d_model]`,
or `[B, T, d_model]` without images. Each token is the sum of its content embedding,
a learned 1D positional embedding, and a learned modality embedding (text ID 0,
image ID 1). Positions start at zero and continue across the whole sequence, so
the first text position is `P` with an image and zero without one. All embedding
tables use standard PyTorch initialization; assembly adds no scaling,
normalization, or dropout.

The actual output length may equal `max_seq_len`; exceeding it raises an error
without truncation. A text-only call can still fit when the configured patch
count exceeds that limit. Inputs must share the module's device, and images
follow normal PyTorch dtype conventions without implicit transfers or casts.
Mask construction remains separate, using the combined sequence length and the
desired image/question prefix length.

```python
from multimodal_loop.model.embeddings import MultimodalEmbedding

embeddings = MultimodalEmbedding(config)
input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
state = embeddings(input_ids, images)

assert state.shape == (2, 19, 64)  # 16 image patches, then 3 text tokens
assert embeddings(input_ids).shape == (2, 3, 64)
```

## Complete model and answer loss

The complete model accepts the same text/image inputs as `MultimodalEmbedding`
and enforces the same combined sequence limit. Runtime recurrence depth defaults
to `config.recurrence_depth` and accepts positive integer overrides. Prelude and
coda stacks may have zero layers. Device, dtype, and dropout behavior follow the
existing components; `.eval()` disables dropout throughout the model.

For causal text-only training, align `logits[:, :-1]` with `input_ids[:, 1:]`.
For question answering, let `P` be the image patch count, `Q` the number of
question tokens, and `T` the total text length. Align
`logits[:, P + Q - 1 : P + T - 1]` with `input_ids[:, Q:]`. This includes the
first answer prediction from the final question position and excludes losses
on visual or question targets. The example uses a shared question length with
`Q >= 1` and at least one answer token per batch item.

```python
import torch
import torch.nn.functional as F

from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer

config = ModelConfig()
model = MultimodalLoopTransformer(config)
input_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

text_logits = model(input_ids)
assert text_logits.shape == (2, 5, config.vocab_size)

images = torch.randn(
    2, config.num_channels, config.image_size, config.image_size, requires_grad=True
)
P, Q, T = config.num_patches, 2, input_ids.shape[1]
mask = build_prefix_mask(P + T, P + Q, device=input_ids.device)
logits = model(input_ids, images, recurrence_depth=3, attention_mask=mask)
assert logits.shape == (2, P + T, config.vocab_size)

answer_logits = logits[:, P + Q - 1 : P + T - 1]
answer_targets = input_ids[:, Q:]
loss = F.cross_entropy(answer_logits.reshape(-1, config.vocab_size), answer_targets.reshape(-1))
loss.backward()
assert images.grad is not None
```

## Minimal training smoke test

Run these commands from the repository root in the installed environment:

```bash
python scripts/train.py
python scripts/train.py --text-only
python scripts/train.py --recurrence-depth 3 --steps 10
```

The script initializes a tiny model and one seeded random tensor batch, then
repeats optimizer updates on that batch. The default multimodal mode has image
patches, three question tokens, and five answer tokens per item. Text-only mode
defaults to causal language modeling. Each run prints its mode, seed, device,
question length, recurrence depth, and each step's loss before the update.
Loss reduction here measures fitting a fixed batch; visual reasoning experiments
are documented separately in the [Milestone 1 results](docs/milestones/milestone1.md).

| Settings | Defaults |
| --- | --- |
| `--steps`, `--batch-size`, `--text-length` | 20, 2, 8 |
| `--question-length` | 3 with images; 0 with `--text-only` |
| `--recurrence-depth`, `--seed`, `--device` | 2, 0, cpu |
| AdamW `--learning-rate`, `--weight-decay` | 0.001, 0.0 |

Model dimensions use `ModelConfig()` defaults. The batch must fit `max_seq_len`,
contain at least two text tokens, and leave at least one target after the
question. `--question-length 0` selects fully causal attention even with images;
a positive value also supports question-answer supervision in text-only mode.
No external dataset or tokenizer is needed.

The reusable loss interface is
`shifted_cross_entropy(logits, input_ids, *, num_image_tokens=0, target_mask=None)`.

It pairs `logits[:, num_image_tokens:-1]` with `input_ids[:, 1:]`. If supplied,
`target_mask` is boolean `[B, T]`, aligned to text token IDs: `True` selects that
token as a target. The helper shifts the mask with the targets, so the first
text token is never supervised. Image logits, the final text logit, and masked
targets contribute no loss. The result is the mean over selected tokens across
the entire batch, and selecting no predictable targets raises an error. All
IDs, including zero, remain ordinary vocabulary IDs; int32 IDs are converted to
int64 for cross-entropy. Supervision masking does not provide padding attention
support.

Use the training loop with an explicitly owned optimizer:

```python
from multimodal_loop.train.trainer import train_on_batch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
losses = train_on_batch(
    model,
    optimizer,
    input_ids,
    images,
    steps=20,
    question_length=2,
    recurrence_depth=3,
)
```

This example uses the model and tensors above. `question_length=0` supervises
all text targets after the first input with causal attention. A positive length
builds an image/question prefix mask and supervises only answers, including the
first answer predicted from the final question input. All batch items share the
same question length. The loop enables training mode and calls
`zero_grad(set_to_none=True)`, forward, backward, and `optimizer.step()` on each
iteration, returning detached Python loss values. A nonfinite loss raises before
backward or an optimizer update. The caller owns tensor placement, optimizer
settings, and RNG state; the loop retains optimizer history across calls.

Recurrence depth is fixed per call, with `None` selecting the model default.
Scheduling, clipping, gradient accumulation, mixed precision, and recurrence
sampling remain subsequent work.

## Checkpoints and deterministic CPU resume

Save after five optimizer updates, then run five additional updates:

```bash
python scripts/train.py --steps 5 --save-checkpoint checkpoints/smoke.pt
python scripts/train.py --resume checkpoints/smoke.pt --steps 5 \
    --save-checkpoint checkpoints/smoke.pt
```

`--steps` always counts updates for the current invocation. In this example,
the resumed run prints steps 6 through 10 and saves `completed_steps=10`.
Saving is opt-in: omitting `--save-checkpoint` leaves the original file unchanged.
The script saves only after all requested updates finish successfully.

On resume, the saved batch, question length, recurrence depth, and optimizer
settings determine the continuation. The script does not regenerate data or
reseed. Explicit new-run settings such as `--seed`, `--text-only`,
`--learning-rate`, or `--recurrence-depth` are rejected with `--resume`.
`--steps`, `--save-checkpoint`, and `--device` remain available. The device must
match the saved backend; the default remains CPU.

Version 2 checkpoints contain:

* Format version, `ModelConfig` values, model dtype and train/eval mode, and
  model weights.
* AdamW state and hyperparameters, completed optimizer steps, the actual text
  and optional image tensors, question length, effective recurrence depth, and
  the original seed when supplied.
* Python `random`, NumPy's global RNG (including its Gaussian cache), and
  PyTorch CPU RNG states, plus the selected CUDA or XLA device RNG state.
* Backend identity and runtime versions for identifying the execution environment.

Existing version 1 CPU checkpoints still load. All saved tensors, including
accelerator weights and optimizer moments, are detached CPU snapshots.

Files contain tensors and ordinary Python values and load with
`torch.load(..., map_location="cpu", weights_only=True)`. Writes use a temporary
file in the destination directory and atomic replacement; a failed write leaves
an existing checkpoint intact. Checkpoints represent completed optimizer-step
boundaries, including step zero. Gradients are omitted because the next training
step clears them.

The public helpers are `save_checkpoint(path, model, optimizer, *,
completed_steps, input_ids, images=None, question_length=0,
recurrence_depth=None, seed=None)` and `load_checkpoint(path, *, device="cpu")`.
The loader returns a `LoadedCheckpoint` with the model, optimizer, completed
steps, batch tensors, question length, effective recurrence depth, and seed.
For example, after creating the file above:

```python
from multimodal_loop.train.checkpoint import load_checkpoint, save_checkpoint
from multimodal_loop.train.trainer import train_on_batch

checkpoint = load_checkpoint("checkpoints/smoke.pt")
losses = train_on_batch(
    checkpoint.model,
    checkpoint.optimizer,
    checkpoint.input_ids,
    checkpoint.images,
    steps=5,
    question_length=checkpoint.question_length,
    recurrence_depth=checkpoint.recurrence_depth,
)
save_checkpoint(
    "checkpoints/continued.pt",
    checkpoint.model,
    checkpoint.optimizer,
    completed_steps=checkpoint.completed_steps + len(losses),
    input_ids=checkpoint.input_ids,
    images=checkpoint.images,
    question_length=checkpoint.question_length,
    recurrence_depth=checkpoint.recurrence_depth,
    seed=checkpoint.seed,
)
```

The loader rebuilds the model in its saved dtype, strictly restores weights and
optimizer state, restores train/eval mode, and restores RNG states last. Resume
training immediately after loading to retain that random sequence.

Exact continuation is tested on CPU float32/float64 in the same software,
hardware, and execution environment. The supported baseline has all parameters
trainable, a uniform train/eval mode, and one AdamW parameter group containing
all model parameters in their original order. Unsupported devices, dtypes,
optimizer layouts, and format versions raise errors. Accelerator checkpoints use
float32 and ordinary AdamW (`foreach=False, fused=False`, with capturable and
differentiable execution disabled). Construct the optimizer after moving the
model to its device. Loading places AdamW moments alongside parameters and keeps
non-capturable step counters on CPU.

Resume on the same backend and hardware/software environment. CUDA indices may
change, but CPU/CUDA/TPU cross-backend continuation is not supported. Accelerator
checks require exact RNG replay and numerical agreement for losses, parameters,
and optimizer moments: `rtol=1e-5, atol=1e-6` on CUDA and `rtol=1e-4, atol=1e-5`
on TPU. These tolerances do not assert bitwise accelerator training equivalence.
Independent random generators, data-loader state, schedulers, and mixed-precision
state remain outside the checkpoint format.

## Single-device CUDA and TPU on Kaggle

Select a GPU or TPU accelerator in Kaggle's notebook settings. The training
script accepts `--device cpu`, `cuda`, `cuda:N`, or `xla` (the first TPU device).
It fails if the requested device is unavailable. There is no automatic CPU
fallback, distributed launch, SPMD, mixed precision, or use of every TPU core.

`train/runtime.py` provides explicit device selection, global host/device
seeding, XLA step synchronization, and device RNG helpers. CPU and CUDA imports
do not import `torch_xla`. Fresh runs construct their model and synthetic batch
on CPU, transfer them, and then construct AdamW. `train_on_batch` keeps caller-owned
placement and calls `torch_xla.sync(wait=True)` after each TPU optimizer update,
as in the [PyTorch/XLA single-device guide](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html).
Supervision masking retains fixed tensor shapes using ignored targets. Existing
value validation remains enabled; host checks can synchronize XLA execution.
This is a correctness path, not a throughput benchmark.

### Install and inspect the environment

Push the desired repository revision before using these cells. In a fresh Kaggle
notebook, clone the branch containing these changes and install the package:

```python
%cd /kaggle/working
!git clone --branch milestone0 https://github.com/Krailon/multi-modal-loop-llm.git
%cd /kaggle/working/multi-modal-loop-llm
%pip install -e ".[dev]"
```

Use the notebook's installed accelerator stack. Do not install this project's
`torch` extra or `requirements.txt` over it. PyTorch/XLA must match PyTorch's
major/minor version, and libtpu must be compatible with that pair. Kaggle's
[TPU image configuration](https://github.com/Kaggle/docker-python/blob/main/tpu/config.txt)
and [TPU Dockerfile](https://github.com/Kaggle/docker-python/blob/main/tpu/Dockerfile)
currently coordinate PyTorch 2.8 with a compatible libtpu; individual notebook
images may differ. These are environment references, not new package pins.
The runtime reports missing or mismatched dependencies without replacing them.

Inspect versions without initializing the TPU in the notebook process:

```python
from importlib.metadata import PackageNotFoundError, version

for package in ("torch", "torch_xla", "libtpu"):
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, "not installed")
```

Run training and tests as subprocesses. For TPU sessions, avoid initializing
JAX, TensorFlow, or XLA in the notebook process first; let each subprocess own
and release the TPU runtime. Restart the notebook session if it already owns it.

### CUDA cells

This uses one GPU, including on Kaggle machines with multiple GPUs:

```bash
%%bash
set -e
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python scripts/train.py --device cuda:0 --text-only --steps 2
python scripts/train.py --device cuda:0 --steps 2 --save-checkpoint /kaggle/working/checkpoints/cuda.pt
python scripts/train.py --device cuda:0 --resume /kaggle/working/checkpoints/cuda.pt --steps 2 --save-checkpoint /kaggle/working/checkpoints/cuda.pt
MULTIMODAL_LOOP_TEST_DEVICE=cuda:0 python -m pytest -q tests/test_accelerators.py
```

### TPU cells

Set PJRT and chip visibility before starting Python. The
[PJRT guide](https://docs.pytorch.org/xla/master/learn/pjrt.html) documents these
single-chip settings; the script selects one XLA device on that chip without
spawning workers. Initial TPU graph compilation takes longer than CPU startup.

```bash
%%bash
set -e
export PJRT_DEVICE=TPU
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0
python -c 'from multimodal_loop.train.runtime import resolve_device, runtime_metadata; print(runtime_metadata(resolve_device("xla")))'
python scripts/train.py --device xla --text-only --steps 2
python scripts/train.py --device xla --steps 2 --save-checkpoint /kaggle/working/checkpoints/xla.pt
python scripts/train.py --device xla --resume /kaggle/working/checkpoints/xla.pt --steps 2 --save-checkpoint /kaggle/working/checkpoints/xla.pt
MULTIMODAL_LOOP_TEST_DEVICE=xla python -m pytest -q tests/test_accelerators.py
```

The hardware suite runs text-only and multimodal cases at depths 1 and 3,
checks prefix isolation and gradients, and compares four uninterrupted updates
with two updates followed by two resumed updates in fresh processes. Dropout is
enabled for resume checks, including exact subsequent random draws and repeated
save neutrality. Checkpoints are compared on CPU. An explicitly requested
unavailable accelerator fails the suite; ordinary `pytest` skips these four
hardware cases. `MULTIMODAL_LOOP_TEST_DEVICE=cpu` can exercise the same subprocess
harness locally, but does not establish CUDA/TPU support.

Record the selected Kaggle accelerator, printed runtime versions, and test output
when running these checks. Single-device CUDA validation has passed in a Kaggle
2×T4 environment, with the user-confirmed result `4 passed in 69.91s (0:01:09)`.
This validates the single-device path on that environment; multi-GPU execution
remains outside the supported scope. TPU support is implemented, with hardware
validation deferred and not required for Milestone 0 sign-off.

Files under `/kaggle/working/checkpoints` must be retained as notebook output
using Kaggle's Save Version workflow before the session ends. In a later session,
attach that output, pass its checkpoint path under `/kaggle/input/...` to
`--resume`, and save the continued run to a new `/kaggle/working/checkpoints/...`
path. Keep the backend and runtime environment consistent when resuming.

## Deterministic single-object color questions

The first synthetic corpus asks **"What color is the object?"** about exactly one
square, circle, or upright triangle. Answers are red, green, blue, or yellow.
Images use a black background and saturated RGB colors (yellow is red + green),
with hard edges and no antialiasing. This is a basic color-perception task;
spatial reasoning and recurrence benefits are later research questions.

`SyntheticShapesConfig` defaults to 32×32 images, object bounding-box sizes
`(8, 12, 16)`, seed 0, and 1,024 training / 256 validation / 256 test examples.
Object sizes must be distinct even integers of at least four pixels, fitting
with a one-pixel canvas margin. Rendering samples pixel centers: squares fill
their bounding boxes, circles are inscribed, and triangles have a top-center
apex and bottom-edge base. Boundary pixels are included when their centers lie
on the shape. Even-sized triangles may leave the top bounding-box row empty.
All rendered tensors are fresh CPU `float32 [3,H,W]` values in `[0,1]`, independent
of PyTorch's default device and dtype.

A layout is `(shape, size, left, top)`. The generator enumerates shape order
`square, circle, triangle`, configured size order, then top and left coordinates,
and shuffles that catalog with a local `random.Random(seed)`. It allocates layouts
without replacement to train, validation, then test. Every allocated layout
appears in all four colors in its own split; labels are therefore balanced even
conditional on shape, size, and position. No shape or color category is reserved
for a held-out split. Very small splits may not sample every shape.

Split sizes count images and must be positive multiples of four. The default
catalog contains 3,345 layouts / 13,380 colored scenes. Requests exceeding the
finite capacity fail. Layout identities and exact rendered images are disjoint
across splits, while individual attributes such as color and size can recur.
This tests unseen layout combinations, not unseen attribute categories.

Within each split, examples are shuffled with a separate local RNG seeded by
`f"{seed}:{split}"`. The same configuration and generator/software version
reproduce ordered scenes and pixels without consuming global Python, NumPy, or
PyTorch RNG state. Changing split sizes or size order can change split membership.
Persist the manifest for an experiment instead of regenerating its splits with
modified settings. Data format/generator version 1 describes these rules.

```python
from multimodal_loop.data.synthetic_shapes import (
    SyntheticShapesConfig,
    build_scene_splits,
    make_example,
)

config = SyntheticShapesConfig()
scenes = build_scene_splits(config)
example = make_example(scenes["train"][0], image_size=config.image_size)
assert example.image.shape == (3, 32, 32)
print(example.question, example.answer, example.scene)
```

`ShapeScene` and the configuration are frozen dataclasses. `build_scene_splits`
returns a dictionary of immutable scene tuples. `render_scene(scene,
image_size=32)` renders one scene; `make_example` adds the fixed question,
one-word answer, and scene metadata. Image tensors remain mutable, but repeated
calls allocate independent storage. Scene metadata and manifest answers are
supervision/inspection information; future model inputs must contain only images
and question tokens (plus appropriately masked answer inputs during training).

Generate the default corpus manifest and inspectable preview from the repo root:

```bash
python scripts/generate_synthetic_data.py --output-dir outputs/synthetic_shapes
```

The script prints answer frequencies and writes `manifest.json` (format version,
configuration, software versions, and ordered scene/question/answer records) plus
`preview.html` (up to 12 examples from each split). Open the HTML in a browser.
It is self-contained and displays horizontal runs of actual raster pixels as
inline SVG, together with questions, answers, and scene metadata. Images are
regenerated from scene descriptions; the manifest does not store image arrays.
No external renderer or new dependency is needed. Reruns replace these two files
in the selected output directory.

Use `--seed`, `--image-size`, `--object-sizes 8 12 16`, `--train-size`,
`--validation-size`, `--test-size`, and `--preview-count` to select another valid
configuration. This command does not tokenize examples or train a model.

## Tokenization and batching for color questions

`ColorQuestionTokenizer` uses vocabulary version 1, independent of dataset
contents or example order:

| ID | Token | ID | Token |
| --- | --- | --- | --- |
| 0 | `What` | 5 | `?` |
| 1 | `color` | 6 | `red` |
| 2 | `is` | 7 | `green` |
| 3 | `the` | 8 | `blue` |
| 4 | `object` | 9 | `yellow` |

It accepts exactly `What color is the object?` and the four lowercase color
answers. It performs no fitting or normalization and has no padding, BOS, EOS,
or unknown-token fallback. `encode_question` returns six IDs, `encode_answer`
returns one ID, and `decode_answer` accepts only IDs 6–9. Evaluation must count
non-color predictions as incorrect rather than decoding them as a valid color.
The model vocabulary must contain at least ten entries; use ten for this task.

`SyntheticColorCollator(ModelConfig)` consumes a nonempty sequence of examples
and returns a `ColorQuestionBatch`. It reads only `image`, `question`, and the
explicit `answer`; it never reads scene metadata or derives labels from image
pixels. This preserves the original labels when images are replaced for future
control experiments. The batch contains no scene descriptions or identifiers.

For B examples and P image patches, the batch holds CPU float32 images
`[B,3,H,W]`, int64 `input_ids[B,7]`, boolean `target_mask[B,7]`, and a shared
boolean `attention_mask[P+7,P+7]`. Images must match the model's configured size
and be finite float32 CPU tensors in `[0,1]`. Stacking allocates fresh storage.
`question_length=6` and `num_image_tokens=P` describe the alignment:

```text
text input:    What color is the object ? red
text position:    0     1  2   3      4 5   6
loss mask:        F     F  F   F      F F   T
```

The bidirectional prefix contains the image patches and all six question tokens.
It cannot attend to the answer input. The last question logit, at combined
position `P+5`, predicts the answer. The answer input has no supervised successor.
Only that one prediction contributes to shifted cross-entropy, averaged across
examples. There is no EOS target.

```python
import torch

from multimodal_loop.data.collator import SyntheticColorCollator
from multimodal_loop.data.synthetic_shapes import (
    SyntheticShapesConfig,
    build_scene_splits,
    make_example,
)
from multimodal_loop.data.text import ColorQuestionTokenizer
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.losses import shifted_cross_entropy
from multimodal_loop.train.trainer import train_on_batch

config = ModelConfig(vocab_size=ColorQuestionTokenizer.vocab_size)
scenes = build_scene_splits(SyntheticShapesConfig())
examples = [make_example(scene) for scene in scenes["train"][:4]]
batch = SyntheticColorCollator(config)(examples)
model = MultimodalLoopTransformer(config)

logits = model(**batch.model_inputs())
loss = shifted_cross_entropy(
    logits,
    batch.input_ids,
    num_image_tokens=batch.num_image_tokens,
    target_mask=batch.target_mask,
)
loss.backward()

# The existing smoke trainer builds equivalent prefix and supervision masks.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = train_on_batch(
    model,
    optimizer,
    batch.input_ids,
    batch.images,
    steps=1,
    question_length=batch.question_length,
)
```

`batch.model_inputs()` returns only `input_ids`, `images`, and `attention_mask`.
`batch.to(device)` returns a new batch with all four tensors transferred together;
initialize accelerators through the existing runtime helper first. Tensor
transfers follow PyTorch storage semantics, so a no-op transfer may share storage.

Evaluation can provide question tokens alone: use `batch.input_ids[:, :6]` with a
fully bidirectional image/question mask of size `P+6`, then read the final logit.
Tests compare this prediction with the answer-supervised sequence and verify
isolation from answer inputs across recurrent depths. The dataset training path
below uses this question-only evaluation protocol.

## Fixed-depth dataset training and validation

`load_synthetic_manifest(path)` validates the saved version, configuration,
record counts, scene bounds, question/answer consistency, all-four-color coverage
per layout, and separation between splits. It preserves the stored record order
and hashes the exact UTF-8 file contents with SHA256. It never regenerates split
membership from the seed. `SyntheticColorDataset(manifest, split)` renders fresh
pixels on demand. Scene metadata stays outside the collated model inputs.

Run the first baseline from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_synthetic.py \
  --manifest outputs/synthetic_shapes/manifest.json \
  --output-dir outputs/color_baseline --epochs 10
```

Defaults are the existing model architecture with vocabulary size 10 and the
manifest's image size, float32, recurrence depth **R=2**, batch size 32, AdamW
learning rate 0.001, weight decay 0, and training seed 0. The default model has
64-wide representations, four attention heads, FFN width 256, one prelude block,
one shared core block, one coda block, 8×8 patches, and dropout 0. Training runs
one optimizer update per batch, with answer-only shifted cross-entropy.
The ten-epoch default corpus budget is **320 updates / 10,240 training examples**.

Each zero-based epoch shuffles every training index exactly once using local
`random.Random(f"{seed}:train:{epoch}")`. Loaders use one process, no workers,
no dropped final batch, and a private PyTorch generator so iterator construction
does not consume dropout randomness. `train_synthetic_epoch` reports the
example-weighted mean of pre-update batch losses, example count, and step count.
Batch size, learning rate, weight decay, seed, and recurrence depth are validated
by `SyntheticTrainingConfig` and exposed as corresponding CLI flags.

`evaluate_synthetic` runs before any updates (epoch 0) and after every epoch at
the same fixed recurrence depth. It forwards images and only the six question
IDs with fully bidirectional prefix attention. The final question logit predicts
the answer. Argmax spans the full vocabulary; non-color predictions count as
incorrect and are reported separately. Validation reports total/correct counts,
accuracy, invalid predictions, and cross-entropy weighted by example count,
including partial batches. Evaluation disables dropout and gradients, preserves
model modes and random streams, and leaves parameters, gradients, and optimizer
state unchanged. The training command never renders or evaluates test examples;
it only validates their manifest metadata for split integrity.

Use `--model-config path/to/model.yaml` for a YAML mapping of `ModelConfig`
overrides. Unspecified fields retain defaults; vocabulary must be 10, images must
match the manifest, and sequence capacity must fit patches plus seven text tokens.
`--recurrence-depth` overrides the runtime depth; otherwise it follows the resolved
model configuration. Both model and training settings are recorded explicitly.
Epoch count is an additional budget for each invocation, not a stopping criterion
based on validation performance. There is no early stopping or best-model selection.

Each output directory contains:

* `settings.json`: resolved model/training settings, tokenizer, manifest hash,
  runtime metadata, and CPU thread count.
* `manifest.json`: the exact corpus used by the run.
* `metrics.json`: epoch 0 and every completed epoch's training/validation metrics.
* `last.pt`: an atomically replaced checkpoint at the last complete epoch boundary
  (also written at epoch 0).

`save_synthetic_checkpoint` and `load_synthetic_checkpoint` use a distinct dataset
format, sharing the existing model/AdamW serialization and atomic-write machinery.
They save configuration, model/optimizer state, mode, backend/runtime, Python,
NumPy, CPU and selected-device RNG state, tokenizer version/vocabulary, manifest
contents/hash, completed epoch/step counts, and metric history. Resume validates
these before restoring randomness, reconstructs the next epoch's shuffle, and
requires the same backend. Exact CPU replay is tested with dropout, partial
batches, and fresh processes; hardware/software and thread settings must match.
Mid-epoch resume is not supported. Existing fixed-batch checkpoints and
`scripts/train.py` retain their original behavior.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_synthetic.py \
  --resume outputs/color_baseline/last.pt \
  --output-dir outputs/color_baseline --epochs 2
```

Resume permits only device, output directory, and additional epoch count; fresh
training/model/manifest overrides are rejected. The embedded manifest permits
resume after the original file disappears. Checkpoint history is authoritative
and repairs stale `metrics.json` output. Fresh runs reject an output directory
that already contains run artifacts.

On Kaggle, after installing the package without replacing its accelerator-specific
PyTorch (see the accelerator setup above), use a writable output directory:

```bash
python scripts/train_synthetic.py --device cuda:0 \
  --manifest /kaggle/working/synthetic_shapes/manifest.json \
  --output-dir /kaggle/working/color_baseline --epochs 10
```

This selects one GPU, including in a 2×T4 runtime. `--device xla` uses the existing
single-device TPU path, whose hardware validation remains deferred. The new dataset
training/resume path is CPU-tested; the earlier Kaggle sign-off covers the
Milestone 0 accelerator infrastructure.

Baseline history and experimental results are recorded in
[Milestone 1 — results and close-out](docs/milestones/milestone1.md).

## Image controls on the same trained model

Evaluate the checkpoint's embedded validation split with correct images, five
predetermined image shuffles (seeds 0–4), and blank images:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/evaluate.py \
  --checkpoint outputs/color_baseline/last.pt \
  --output-dir outputs/color_controls
```

The command uses the checkpoint's saved runtime recurrence depth and batch size
(R=2 and 32 for this baseline), model weights, and corpus. It performs no training,
selects no checkpoint, and defaults to validation. Use `--split test` explicitly
for a frozen test evaluation; only the selected split is rendered and evaluated. The existing
same-backend checkpoint restriction applies; `--device cuda:0` evaluates a CUDA
checkpoint on one GPU. This increment was validated on CPU.

`evaluate_image_controls(model, dataset, recurrence_depth=..., batch_size=32,
shuffle_seeds=(0, 1, 2, 3, 4))` reuses question-only evaluation and replaces only
the completed batch's image tensor. Recipient questions, original answer targets,
example order, attention masks, image-token counts, positional/modality embeddings,
and recurrence depth remain identical between conditions. Scene metadata is never
passed to the model. Blank images are all-zero tensors with the same shape and
dtype, so image patch tokens remain present; blank does not mean `images=None`.

Each shuffle permutes the entire selected split with local
`random.Random(seed).shuffle`, independently of batch size. Every donor image is
used exactly once. Self-pairings and same-color pairings are retained, with no
label-based filtering. Donor answers never replace recipient targets. Original
answer fields are used only to report the fraction of same-color pairings after
the permutation is fixed. Ordinary random pairings have expected color agreement
of 25% on this balanced corpus, with finite-sample variation. Blank images also
remove the object entirely, so the shuffled control additionally checks performance
when real corpus images remain present but their pairing is broken.

Use `--batch-size` to override evaluation batching and `--shuffle-seeds 0 1 2 3 4`
to specify distinct integer seeds. All examples are included, even with a partial
last batch. Model modes, parameters, gradients, optimizer state, and random streams
are preserved during evaluation. The CLI loads training RNG state through the
existing checkpoint loader; it never updates the saved checkpoint or training
metrics. `--split` accepts `validation` (default) or `test`; there is no recurrence
override. Donor images always come exclusively from the selected split.

The printed table includes counts, accuracy, cross-entropy, and invalid predictions
for every condition. `controls.json` contains these metrics, every donor permutation
(recipient index → donor index in manifest order), seeds and same-color pairing
fractions, shuffled mean/min/max accuracy and loss, and correct-minus-control
accuracy gaps expressed as fractions. It also records checkpoint/manifest SHA256,
training progress, model/training/evaluation settings, runtime, and CPU thread count.
An existing report is rejected; choose a new output directory for another run.
The five shuffle results describe pairing variation on one validation set, not five
independent datasets or independent training runs.

Validation and frozen-test results, full protocols, and limitations are recorded in
[Milestone 1 — results and close-out](docs/milestones/milestone1.md).

## Three-object rows and relational color answers

The first Milestone 2 data increment supports handcrafted horizontal rows of
exactly three objects. `MultiObjectScene` contains an immutable tuple of existing
`ShapeScene` objects and an `image_size` (default 32). Bounding-box vertical centers
must align, adjacent boxes must have at least one blank pixel between them, and
all boxes must retain a one-pixel canvas margin. Existing even-size and shape/color
validation applies. Tuple order is arbitrary; spatial order comes from coordinates.
Repeated shapes and colors are allowed in a scene.

`RelationalColorQuestion(anchor_shape, direction)` accepts a shape and `left` or
`right`. Its `.text` asks for the color of the object immediately beside that
anchor. The anchor must occur exactly once, and the requested neighbor must exist;
otherwise resolution raises `ValueError`. With two objects on the requested side,
the immediate neighbor in horizontal order supplies the answer.

```python
from multimodal_loop.data.relational_shapes import (
    MultiObjectScene,
    RelationalColorQuestion,
    make_relational_example,
)
from multimodal_loop.data.synthetic_shapes import ShapeScene

scene = MultiObjectScene(
    objects=(
        ShapeScene("circle", "green", left=1, top=12, size=8),
        ShapeScene("square", "red", left=12, top=12, size=8),
        ShapeScene("triangle", "blue", left=23, top=12, size=8),
    )
)
left = make_relational_example(scene, RelationalColorQuestion("square", "left"))
right = make_relational_example(scene, RelationalColorQuestion("square", "right"))
assert left.answer == "green" and right.answer == "blue"
assert left.image.equal(right.image)  # Same pixels, different questions and answers.
print(left.question)  # What color is the object immediately left of the square?
```

`render_multi_object_scene` combines the existing renderer's disjoint objects into
fresh CPU float32 `[3,H,W]` pixels, independently of default tensor device/dtype,
without consuming global randomness. `resolve_relation` returns the target object;
`answer_relational_question` returns its color without rendering. The frozen
`RelationalExample` holds pixels, question text, answer, scene, and query metadata;
its tensor remains mutable and independently allocated for each example. Scene
and query metadata are supervision/inspection information, never model inputs.

These helpers establish geometry and answer semantics. The corpus generator below
adds balancing and split separation, and the loading/batching path below prepares
relational examples for model input. The relational training and evaluation path
is documented below. The Milestone 1 tokenizer/collator still supports only
its original fixed question; existing formats and experiments remain unchanged.

## Balanced relational corpus generation

`RelationalCorpusConfig` and `build_relational_splits(config)` in
`multimodal_loop.data.relational_corpus` generate metadata for a finite corpus.
Defaults are 32×32 images, object sizes `(6, 8)`, seed 0, and geometry counts
**16 training / 4 validation / 4 test**. Counts refer to geometries, not images or
question-answer examples. Every generated image contains one square, one circle,
and one triangle with three distinct colors selected from red, green, blue, yellow.

A geometry is the spatially ordered tuple of three `(left, top, size)` bounding
boxes, excluding shapes and colors. The generator enumerates feasible size triples
in configured order, followed by ascending vertical centers, left margins, first
internal gaps, and second internal gaps. The right margin is the remaining space.
All margins and internal gaps are at least one pixel. Size triples that cannot fit
are skipped. The default finite catalog contains **25,136 geometries**; `.capacity`
counts these without constructing the catalog. Invalid settings or requested
geometry totals exceeding capacity fail before records are expanded or files written.

A local `random.Random(seed)` shuffles the catalog, then geometries are allocated
without replacement to train, validation, and test. **Every shape/color variant
and every question for a geometry remains in the same split.** Evaluation therefore
holds out bounding-box arrangements rather than only recolorings or shape orders.
Changing corpus settings can change membership; preserve the generated manifest
for an experiment instead of regenerating with modified settings.

Each geometry expands into all six shape orders and all 24 ordered assignments of
three distinct colors: **144 images**. Every image has four valid questions, ordered
by spatial anchor position and then left/right direction, skipping missing neighbors.
These target the middle, left, right, and middle objects respectively. Each image
record retains all four queries, exact question texts, and explicit answers together.
Objects are stored in canonical left-to-right order; no raster arrays are stored.
Image records are shuffled within each split with local
`random.Random(f"{seed}:{split}")`. Generation consumes no global RNG state.

| Split | Geometries | Images | Question-answer examples | Examples per answer color |
| --- | ---: | ---: | ---: | ---: |
| Training | 16 | 2,304 | 9,216 | 2,304 |
| Validation | 4 | 576 | 2,304 | 576 |
| Test | 4 | 576 | 2,304 | 576 |

Answers are exactly balanced both overall and conditional on each of the six
question texts, even within one geometry's expansion. Distinct colors ensure
that questions targeting different objects have different answers. However, two
of the four questions target the middle object: **an image-only strategy always
answering with the middle object's color scores 50%**, not 25%. Later learning
experiments must test question dependence against this possibility. These are
dataset properties, not learned-model results.

```bash
python scripts/generate_relational_data.py --output-dir outputs/relational_shapes
```

The command prints geometry, image, question-answer, and answer-frequency counts.
It writes a distinct `relational_color_rows` version-1 `manifest.json` with config,
software versions, and ordered split records. Each record contains `scene` and
`questions`; each question contains structured `query`, exact `question` text,
and `answer`. The API returns tuples of frozen `RelationalSceneRecord` and
`RelationalQA` records. Scene/query metadata is supervision and inspection data,
never model input. This format does not change or replace the Milestone 1 manifest.

`preview.html` is self-contained and displays actual raster pixels with all four
questions/answers for up to 12 images per split. The relational and original
single-object generators share the same raster-to-SVG helper. Use `--preview-count`
to change the number of preview images. Reruns replace the manifest and preview
in the chosen directory; generated files under `outputs/` are ignored by git.
Previewing test scenes is data inspection, not model evaluation.

Use `--image-size`, `--object-sizes`, `--seed`, `--train-geometry-count`,
`--validation-geometry-count`, and `--test-geometry-count` to configure another
valid corpus. All three geometry counts must be positive integers. Object sizes
must be distinct even integers of at least four, fitting the canvas individually;
the total request must fit the feasible three-object catalog. This command writes
metadata and previews only; use the separate loading/batching interfaces below to
prepare tensors. It does not train or evaluate a model.

## Loading and batching relational examples

`load_relational_manifest(path)` and `parse_relational_manifest(content)` in
`multimodal_loop.data.relational_dataset` validate the stored corpus without
rendering or regenerating it from the seed. Validation covers kind/version,
configuration, image size, canonical spatial object order, scene geometry,
configured object sizes, distinct shapes/colors, and all four valid queries with
exact question text and resolved answers. Every geometry must retain all six
shape orders and 24 color assignments, with no duplicate variants or geometry
overlap between splits. Both image-record and question order are preserved as
stored. The frozen `RelationalManifest` retains immutable grouped records, exact
UTF-8 content, and its SHA256; loading does not consume global randomness.

`RelationalColorDataset(manifest, split)` flattens images then their stored
questions: index `i` selects image `i // 4` and question `i % 4`. Default lengths
are 9,216 training and 2,304 per held-out split. `.records` retains the grouped
records for future paired-question evaluation. Each access renders fresh CPU
float32 pixels without caching and returns the stored text and explicit answer
along with scene/query metadata. Negative indices follow ordinary sequence
semantics; indices outside the dataset raise `IndexError`.

`RelationalColorTokenizer` is a separate version-1 task tokenizer. It preserves
Milestone 1 IDs 0–9 and appends:

| ID | Token | ID | Token |
| --- | --- | --- | --- |
| 10 | immediately | 14 | square |
| 11 | left | 15 | circle |
| 12 | right | 16 | triangle |
| 13 | of | | |

Only the six exact relational question strings are accepted. Questions have
**11 tokens**, including two occurrences of `the`. Answers still use IDs **6–9**;
answer-ID validity is independent of question length. Other answer predictions
are rejected by `decode_answer`. There is no normalization, fitting, padding,
or special-token insertion. For example:

```text
What color is the object immediately left of the square ?
   0     1  2   3      4          10   11 13   3     14 5
```

`RelationalColorCollator(ModelConfig)` reads only `image`, `question`, and explicit
`answer`, never scene/query metadata. It returns the existing `ColorQuestionBatch`
with images `[B,3,H,W]`, int64 IDs `[B,12]`, a boolean target mask selecting only the
last answer, and a shared prefix-attention mask. With P image patches, the first
P+11 positions are bidirectional and cannot attend to the answer input. The logit
at position **P+10** predicts the answer; the final answer input has no supervised
successor. The model requires three channels, vocabulary size at least 17, and
capacity for P+12 positions. Use vocabulary size 17 for the initial relational task.

```python
from multimodal_loop.data.collator import RelationalColorCollator
from multimodal_loop.data.relational_dataset import (
    RelationalColorDataset,
    load_relational_manifest,
)
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.model.config import ModelConfig

manifest = load_relational_manifest("outputs/relational_shapes/manifest.json")
dataset = RelationalColorDataset(manifest, "train")
config = ModelConfig(
    vocab_size=RelationalColorTokenizer.vocab_size,
    image_size=manifest.config.image_size,
)
batch = RelationalColorCollator(config)([dataset[i] for i in range(4)])
assert batch.input_ids.shape == (4, 12)
assert batch.question_length == 11
# These four questions share identical image pixels.
assert batch.images[0].equal(batch.images[3])
```

Use `batch.model_inputs()` for the forward arguments, `.to(device)` to transfer
tensors together, and `shifted_cross_entropy` with `batch.num_image_tokens` and
`batch.target_mask` for answer-only loss. Question-only forward passes use the
first 11 IDs, an all-prefix mask of size P+11, and the final logit. Tests compare
these predictions to the answer-supervised sequence at multiple recurrent depths
and verify isolation from changed answer inputs and successful gradient flow.

The shared batch helpers preserve Milestone 1 behavior. The existing synthetic
training/evaluation CLIs, dataset checkpoint format, and control evaluator remain
specific to the original six-token color question. The relational task uses
separate entry points and a distinct checkpoint kind. A shared tokenizer version
number or shared color IDs does not make the two task checkpoints interchangeable.

## Fixed-depth relational training and controls

Import the [CUDA smoke notebook](notebooks/kaggle_relational_smoke.ipynb), then the
[Milestone 2 baseline notebook](notebooks/kaggle_milestone2_baseline.ipynb), into
Kaggle from GitHub. They provide setup, training/resume, validation controls, and
artifact downloads. See [notebook import and usage](docs/relational_training.md#importable-notebooks).

`scripts/train_relational.py` trains manifest-backed relational QA examples with
answer-only loss and epoch-boundary resume. `scripts/evaluate_relational.py`
evaluates a frozen checkpoint with correct, blank, and shuffled images, plus
within-image shuffled questions. Reports include per-question accuracy, all-four
accuracy, and different-answer-pair accuracy: always predicting the middle color
can reach 50% overall while scoring zero on both grouped metrics.

See [relational training, checkpoint, and control usage](docs/relational_training.md)
for Kaggle commands, artifact formats, and interpretation. The
[Milestone 2 protocol and future results](docs/milestones/milestone2.md) record
the agreed 10-epoch, R=2 baseline, its results, and validation gates. The first
baseline passed two of six gates. Its frozen diagnosis is recorded in the same
results document. Use the
[geometry-diversity notebook](notebooks/kaggle_milestone2_geometry_diversity.ipynb)
for the [next controlled experiment](docs/milestones/milestone2_geometry_diversity.md),
following the [direct-grounding results](docs/milestones/milestone2_shape_grounding.md#results).

Immediate objective — training geometry diversity:

> Expand direct-grounding training to 128 layouts at the same update budget.
> Preserve the architecture, renderer, and recorded validation/test scenes.

Scale comes later.

First, build and measure useful capabilities. Investigate where recurrence helps
alongside that progress.
