"""Coverage-driven training geometry expansion with unchanged held-out records."""

import json
import random
from collections import Counter
from dataclasses import asdict, replace
from itertools import permutations

from multimodal_loop.data.relational_corpus import (
    Geometry,
    RelationalQA,
    RelationalSceneRecord,
    _geometry_catalog,
)
from multimodal_loop.data.relational_dataset import RelationalManifest, parse_relational_manifest
from multimodal_loop.data.relational_shapes import (
    MultiObjectScene,
    RelationalColorQuestion,
    answer_relational_question,
)
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES, ShapeScene, _integer


def geometry_of(record) -> Geometry:
    return tuple((o.left, o.top, o.size) for o in record.scene.objects)


def _sizes(geometry):
    return tuple(o[2] for o in geometry)


def _features(geometry, patch_size):
    coordinates = {
        (slot, size, axis, value)
        for slot, (x, y, size) in enumerate(geometry)
        for axis, value in (("x", x), ("y", y))
    }
    offsets = {
        (slot, size, x % patch_size, y % patch_size) for slot, (x, y, size) in enumerate(geometry)
    }
    return coordinates, offsets


def select_training_geometries(source, *, geometry_count=128, seed=0, patch_size=8):
    """Greedy marginal-coordinate coverage, then joint patch-offset coverage.

    Ties use a seeded shuffle of the canonical catalog, never set iteration order.
    Size-triple frequencies stay exactly proportional to the original training set.
    """
    _integer("geometry_count", geometry_count, 1)
    _integer("seed", seed)
    _integer("patch_size", patch_size, 1)
    original = {geometry_of(r) for r in source.splits["train"]}
    excluded = {geometry_of(r) for s in ("validation", "test") for r in source.splits[s]}
    if geometry_count < len(original) or geometry_count % len(original):
        raise ValueError("geometry_count must be an integer multiple of the original count")
    multiplier = geometry_count // len(original)
    original_counts = Counter(_sizes(g) for g in original)
    remaining = {sizes: (multiplier - 1) * n for sizes, n in original_counts.items()}
    catalog = [
        g
        for g in _geometry_catalog(source.config)
        if g not in excluded and _sizes(g) in original_counts
    ]
    available = Counter(_sizes(g) for g in catalog if g not in original)
    if any(available[s] < n for s, n in remaining.items()):
        raise ValueError("not enough eligible geometries for the size-triple quotas")
    random.Random(seed).shuffle(catalog)
    features = {g: _features(g, patch_size) for g in catalog}
    selected = set(original)
    covered, offsets = set(), set()
    for g in original:
        a, b = features[g]
        covered.update(a)
        offsets.update(b)
    initial = (set(covered), set(offsets))
    for _ in range(geometry_count - len(original)):
        best = max(
            (g for g in catalog if g not in selected and remaining[_sizes(g)]),
            key=lambda g: (len(features[g][0] - covered), len(features[g][1] - offsets)),
        )
        selected.add(best)
        remaining[_sizes(best)] -= 1
        a, b = features[best]
        covered.update(a)
        offsets.update(b)
    universe = (
        set().union(*(a for a, _ in features.values())),
        set().union(*(b for _, b in features.values())),
    )
    coverage = {}
    for i, (name, result) in enumerate((("coordinates", covered), ("patch_offsets", offsets))):
        coverage[name] = {
            "original": len(initial[i]),
            "selected": len(result),
            "attainable": len(universe[i]),
            "original_features": sorted(initial[i]),
            "selected_features": sorted(result),
            "missing_features": sorted(universe[i] - result),
        }
    coverage["size_triples"] = [
        {"sizes": sizes, "original": n, "selected": n * multiplier}
        for sizes, n in sorted(original_counts.items())
    ]
    return sorted(selected), coverage


def _expand(geometries, image_size):
    records = []
    for geometry in geometries:
        for shapes in permutations(SHAPES):
            for colors in permutations(COLORS, 3):
                scene = MultiObjectScene(
                    tuple(
                        ShapeScene(shape, color, x, y, size)
                        for shape, color, (x, y, size) in zip(shapes, colors, geometry, strict=True)
                    ),
                    image_size=image_size,
                )
                questions = []
                for index, obj in enumerate(scene.objects):
                    for direction, target in (("left", index - 1), ("right", index + 1)):
                        if 0 <= target < 3:
                            query = RelationalColorQuestion(obj.shape, direction)
                            questions.append(
                                RelationalQA(
                                    query, query.text, answer_relational_question(scene, query)
                                )
                            )
                records.append(RelationalSceneRecord(scene, tuple(questions)))
    return records


def derive_geometry_corpus(source: RelationalManifest, *, geometry_count=128, seed=0, patch_size=8):
    """Return a validated derived manifest and self-contained derivation provenance.

    Source software fields and held-out JSON records are preserved. Selection and
    record ordering are deterministic across machines; no runtime version is added.
    The source manifest config alone cannot regenerate this derived corpus.
    """
    geometries, coverage = select_training_geometries(
        source, geometry_count=geometry_count, seed=seed, patch_size=patch_size
    )
    records = _expand(geometries, source.config.image_size)
    random.Random(f"{seed}:train").shuffle(records)
    original = json.loads(source.content)
    payload = {
        **original,
        "config": asdict(replace(source.config, train_geometry_count=geometry_count)),
        "splits": {**original["splits"], "train": [asdict(r) for r in records]},
    }
    content = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    manifest = parse_relational_manifest(content)
    # JSON-native metadata is identical before and after checkpoint/JSON round trips.
    provenance = json.loads(
        json.dumps(
            {
                "kind": "training_geometry_diversity",
                "version": 1,
                "source_manifest_content": source.content,
                "source_manifest_sha256": source.sha256,
                "manifest_sha256": manifest.sha256,
                "geometry_count": geometry_count,
                "selection_seed": seed,
                "patch_size": patch_size,
                "selected_geometries": geometries,
                "coverage": coverage,
            }
        )
    )
    return manifest, provenance


def validate_geometry_derivation(manifest, provenance):
    """Reproduce the derivation instead of accepting a caller-supplied hash alone."""
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "training_geometry_diversity"
        or type(provenance.get("version")) is not int
        or provenance["version"] != 1
    ):
        raise ValueError("invalid geometry derivation provenance")
    try:
        source = parse_relational_manifest(provenance["source_manifest_content"])
        derived, expected = derive_geometry_corpus(
            source,
            geometry_count=provenance["geometry_count"],
            seed=provenance["selection_seed"],
            patch_size=provenance["patch_size"],
        )
        if manifest.content != derived.content or provenance != expected:
            raise ValueError("manifest or provenance disagrees with deterministic derivation")
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete geometry derivation provenance") from error
    return source


def presentation_summary(manifest, *, max_steps=2880, batch_size=32, seed=0):
    """Inspect the exact training index stream, without rendering or model inference."""
    _integer("max_steps", max_steps, 1)
    _integer("batch_size", batch_size, 1)
    _integer("seed", seed)
    count = 3 * len(manifest.splits["train"])
    visits = [0] * count
    steps = epoch = 0
    while steps < max_steps:
        indices = list(range(count))
        random.Random(f"{seed}:train:{epoch}").shuffle(indices)
        for start in range(0, count, batch_size):
            if steps == max_steps:
                break
            for index in indices[start : start + batch_size]:
                visits[index] += 1
            steps += 1
        epoch += 1
    geometries = sorted({geometry_of(r) for r in manifest.splits["train"]})
    ids = {g: f"train:g{i:03d}" for i, g in enumerate(geometries)}
    groups = {name: Counter() for name in ("geometry", "shape", "color", "size")}
    for i, n in enumerate(visits):
        record = manifest.splits["train"][i // 3]
        shape = SHAPES[i % 3]
        obj = next(o for o in record.scene.objects if o.shape == shape)
        for name, key in (
            ("geometry", ids[geometry_of(record)]),
            ("shape", shape),
            ("color", obj.color),
            ("size", str(obj.size)),
        ):
            groups[name][key] += n
    return {
        "max_steps": max_steps,
        "batch_size": batch_size,
        "seed": seed,
        "qa_count": count,
        "unique_qas_seen": sum(n > 0 for n in visits),
        "presentations": sum(visits),
        "mean_presentations_per_qa": sum(visits) / count,
        "qa_visit_histogram": [
            {"visits": n, "qa_count": total} for n, total in sorted(Counter(visits).items())
        ],
        "exposure_totals": {k: dict(sorted(v.items())) for k, v in groups.items()},
    }
