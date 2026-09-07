"""Write a balanced relational corpus manifest and grouped-question preview."""

import argparse
import html
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.preview import raster_svg
from multimodal_loop.data.relational_corpus import (
    CORPUS_KIND,
    FORMAT_VERSION,
    RelationalCorpusConfig,
    build_relational_splits,
)
from multimodal_loop.data.relational_shapes import render_multi_object_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = RelationalCorpusConfig()
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--object-sizes", nargs="+", type=int, default=defaults.object_sizes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    for split in ("train", "validation", "test"):
        parser.add_argument(
            f"--{split}-geometry-count",
            type=int,
            default=getattr(defaults, f"{split}_geometry_count"),
        )
    parser.add_argument("--preview-count", type=int, default=12, help="Maximum images per split.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/relational_shapes"))
    args = parser.parse_args()
    try:
        config = RelationalCorpusConfig(
            image_size=args.image_size,
            object_sizes=tuple(args.object_sizes),
            seed=args.seed,
            train_geometry_count=args.train_geometry_count,
            validation_geometry_count=args.validation_geometry_count,
            test_geometry_count=args.test_geometry_count,
        )
        if args.preview_count <= 0:
            raise ValueError("preview-count must be positive")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    splits = build_relational_splits(config)
    manifest = {
        "kind": CORPUS_KIND,
        "format_version": FORMAT_VERSION,
        "config": asdict(config),
        "software": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "splits": {
            split: [asdict(record) for record in records] for split, records in splits.items()
        },
    }
    page = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Relational color questions</title><style>",
        "body{font:16px system-ui;margin:2rem;background:#f4f4f4;color:#222}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}",
        "figure{margin:0;padding:1rem;background:white}svg{width:100%;max-width:256px}",
        "figcaption{margin-top:.6rem}code{font-size:.8rem}</style><body>",
        "<h1>Relational color questions</h1>",
        f"<p>Seed {config.seed}; three distinct shapes and colors per image.</p>",
    ]
    for split, records in splits.items():
        counts = dict(
            sorted(Counter(qa.answer for record in records for qa in record.questions).items())
        )
        geometries = getattr(config, f"{split}_geometry_count")
        print(
            f"{split}: {geometries} geometries; {len(records)} images; "
            f"{len(records) * 4} QA examples; answers={counts}"
        )
        page.append(f'<h2>{split} ({len(records)} images)</h2><div class="grid">')
        for index, record in enumerate(records[: args.preview_count]):
            page.append(f'<figure data-split="{split}" data-index="{index}">')
            page.append(raster_svg(render_multi_object_scene(record.scene)))
            page.append("<figcaption><ol>")
            for qa in record.questions:
                page.append(
                    f"<li>{html.escape(qa.question)} <strong>{html.escape(qa.answer)}</strong></li>"
                )
            page.append(
                "</ol><code>" + html.escape(str(record.scene)) + "</code></figcaption></figure>"
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
