"""Derive broader-coverage training scenes while retaining source validation/test records."""

import argparse
from pathlib import Path

from multimodal_loop.data.geometry_diversity import derive_geometry_corpus, presentation_summary
from multimodal_loop.data.relational_dataset import load_relational_manifest
from multimodal_loop.eval.baseline_artifacts import BASELINE_MANIFEST_SHA256
from multimodal_loop.eval.shape_grounding_run import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-count", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("generation requires a fresh output directory")
        source = load_relational_manifest(args.source_manifest)
        if not args.smoke and (
            source.sha256 != BASELINE_MANIFEST_SHA256 or args.geometry_count != 128
        ):
            raise ValueError("research generation requires the original source and 128 geometries")
        manifest, provenance = derive_geometry_corpus(source, geometry_count=args.geometry_count)
        coverage = provenance["coverage"]
        if not args.smoke and (
            coverage["coordinates"]["selected"] != 204
            or coverage["coordinates"]["missing_features"]
            or coverage["patch_offsets"]["selected"] != 329
        ):
            raise ValueError("generated coverage disagrees with the fixed protocol")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "source_manifest.json").write_bytes(source.content.encode())
    (args.output_dir / "manifest.json").write_bytes(manifest.content.encode())
    write_json(args.output_dir / "derivation.json", provenance)
    write_json(args.output_dir / "coverage.json", coverage)
    write_json(
        args.output_dir / "presentations.json",
        {
            "reference": presentation_summary(source),
            "derived": presentation_summary(manifest),
        },
    )
    print(f"Generated {len(manifest.splits['train'])} training images; SHA256={manifest.sha256}")


if __name__ == "__main__":
    main()
