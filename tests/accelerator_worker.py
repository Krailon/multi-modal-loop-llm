"""Fresh-process hardware checks, launched by test_accelerators.py."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import load_checkpoint, save_checkpoint
from multimodal_loop.train.runtime import (
    capture_device_rng,
    finish_step,
    resolve_device,
    restore_device_rng,
    runtime_metadata,
    seed_everything,
)
from multimodal_loop.train.trainer import train_on_batch


def draws(device):
    values = torch.rand(12, device=device)
    finish_step(device)
    return {
        "device": values.cpu(),
        "cpu": torch.rand(12),
        "python": random.random(),
        "numpy": np.random.random(12).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    print(runtime_metadata(device), flush=True)
    if args.resume:
        restored = load_checkpoint(args.resume, device=device)
        model, optimizer = restored.model, restored.optimizer
        ids, images = restored.input_ids, restored.images
        completed = restored.completed_steps
        depth = restored.recurrence_depth
        # AdamW moments belong to parameters; non-capturable step counters stay CPU.
        for parameter, state in optimizer.state.items():
            assert parameter.device == device
            if state:
                assert state["step"].device.type == "cpu"
                assert state["exp_avg"].device == device
                assert state["exp_avg_sq"].device == device
    else:
        seed_everything(37, device)
        config = ModelConfig(
            vocab_size=16, d_model=8, n_heads=2, d_ff=16, image_size=4, patch_size=2, dropout=0.25
        )
        model = MultimodalLoopTransformer(config).to(device)
        ids = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
        images = torch.randn(1, 3, 4, 4).to(device) if args.images else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
        completed, depth = 0, args.depth
        # Prefix states must remain independent of future answer inputs even
        # after recurrence. Compare equal-shaped forwards with different answers.
        model.eval()
        patches = config.num_patches if images is not None else 0
        mask = build_prefix_mask(patches + ids.shape[1], patches + 2, device=device)
        changed = torch.tensor([[0, 1, 8, 9, 10]]).to(device)
        before = tuple(id(p) for p in model.parameters())
        if images is not None:
            images.requires_grad_()
        logits = model(ids, images, recurrence_depth=depth, attention_mask=mask)
        other = model(changed, images, recurrence_depth=depth, attention_mask=mask)
        assert logits.shape == (1, patches + 5, 16)
        torch.testing.assert_close(
            logits[:, : patches + 2].cpu(), other[:, : patches + 2].cpu(), rtol=0, atol=0
        )
        logits[:, patches + 1].square().mean().backward()
        finish_step(device)
        assert before == tuple(id(p) for p in model.parameters())
        assert any(
            p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.core.parameters()
        )
        assert all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
        )
        if images is not None:
            assert images.grad is not None and bool(images.grad.abs().sum() > 0)
            images = images.detach()
        model.train()

    before = model.lm_head.weight.detach().cpu().clone()
    losses = (
        train_on_batch(
            model,
            optimizer,
            ids,
            images,
            steps=args.steps,
            question_length=2,
            recurrence_depth=depth,
        )
        if args.steps
        else []
    )
    if args.steps:
        assert not torch.equal(before, model.lm_head.weight.detach().cpu())
        assert all(np.isfinite(loss) for loss in losses)
    # Predict next draws before saving, then rewind. Saving twice must be neutral,
    # including any XLA barriers used to materialize checkpoint tensors.
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        capture_device_rng(device),
    )
    expected = draws(device)
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    restore_device_rng(state[3], device)
    for _ in range(2):
        save_checkpoint(
            args.output,
            model,
            optimizer,
            completed_steps=completed + args.steps,
            input_ids=ids,
            images=images,
            question_length=2,
            recurrence_depth=depth,
            seed=37,
        )
    actual = draws(device)
    for key in ("cpu", "device"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["python"] == expected["python"]
    assert actual["numpy"] == expected["numpy"]
    torch.save({"losses": losses, "draws": actual}, args.output.with_suffix(".results.pt"))


if __name__ == "__main__":
    main()
