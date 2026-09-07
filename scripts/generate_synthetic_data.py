"""Write a reproducible scene manifest and a self-contained raster preview."""

import argparse
import html
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.synthetic_shapes import (
    FORMAT_VERSION,
    QUESTION,
    SyntheticShapesConfig,
    build_scene_splits,
    make_example,
)


def raster_svg(image: torch.Tensor) -> str:
    """Encode the actual tensor as horizontal pixel runs, without image libraries."""
    pixels = (image * 255).to(torch.uint8).permute(1, 2, 0).tolist()
    height, width = image.shape[1:]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Synthetic object image" shape-rendering="crispEdges">',
        f'<rect width="{width}" height="{height}" fill="#000000"/>',
    ]
    for y, row in enumerate(pixels):
        left = 0
        while left < width:
            right = left + 1
            while right < width and row[right] == row[left]:
                right += 1
            if any(row[left]):
                color = "#" + "".join(f"{channel:02x}" for channel in row[left])
                parts.append(
                    f'<rect x="{left}" y="{y}" width="{right - left}" height="1" fill="{color}"/>'
                )
            left = right
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = SyntheticShapesConfig()
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--object-sizes", nargs="+", type=int, default=defaults.object_sizes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--train-size", type=int, default=defaults.train_size)
    parser.add_argument("--validation-size", type=int, default=defaults.validation_size)
    parser.add_argument("--test-size", type=int, default=defaults.test_size)
    parser.add_argument("--preview-count", type=int, default=12, help="Maximum examples per split.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_shapes"))
    args = parser.parse_args()
    try:
        config = SyntheticShapesConfig(
            image_size=args.image_size,
            object_sizes=tuple(args.object_sizes),
            seed=args.seed,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
        )
        if args.preview_count <= 0:
            raise ValueError("preview-count must be positive")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    splits = build_scene_splits(config)
    manifest = {
        "format_version": FORMAT_VERSION,
        "config": asdict(config),
        "software": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "splits": {
            split: [
                {"scene": asdict(scene), "question": QUESTION, "answer": scene.color}
                for scene in scenes
            ]
            for split, scenes in splits.items()
        },
    }
    page = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Synthetic color questions</title><style>",
        "body{font:16px system-ui;margin:2rem;background:#f4f4f4;color:#222}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}",
        "figure{margin:0;padding:1rem;background:white}svg{width:100%;max-width:256px}",
        "figcaption{margin-top:.6rem}code{font-size:.8rem}</style><body>",
        "<h1>Synthetic color questions</h1>",
        f"<p>Seed: {config.seed}. Preview of deterministic, disjoint layout splits.</p>",
    ]
    for split, scenes in splits.items():
        counts = dict(sorted(Counter(scene.color for scene in scenes).items()))
        print(f"{split}: {len(scenes)} examples; answers={counts}")
        page.append(f'<h2>{split} ({len(scenes)} examples)</h2><div class="grid">')
        for index, scene in enumerate(scenes[: args.preview_count]):
            example = make_example(scene, image_size=config.image_size)
            page.append(
                f'<figure data-split="{split}" data-index="{index}">'
                + raster_svg(example.image)
                + "<figcaption>"
                + html.escape(example.question)
                + "<br>Answer: <strong>"
                + html.escape(example.answer)
                + "</strong><br><code>"
                + html.escape(str(scene))
                + "</code></figcaption></figure>"
            )
        page.append("</div>")
    page.append("</body></html>")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "preview.html").write_text("\n".join(page) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'manifest.json'} and {args.output_dir / 'preview.html'}")


if __name__ == "__main__":
    main()
