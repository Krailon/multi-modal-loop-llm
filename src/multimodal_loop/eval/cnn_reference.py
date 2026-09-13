"""Audit the completed four-arrangement predictions without loading reference weights."""

import json
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from multimodal_loop.data.multi_arrangement import parse_multi_arrangement
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.multi_arrangement import aggregate_role
from multimodal_loop.train.cnn_checkpoint import MANIFEST_SHA256

REFERENCE_HASHES = {
    "data/manifest.json": MANIFEST_SHA256,
    "diagnosis/summary.json": "367b04f0c48120eea105b90d1664906e1a38890849dcc732397177555aee1970",
    "diagnosis/predictions.jsonl": (
        "1d616e65d1721b726e287d19b9d6b08040cd28860958e962a842b978c55239c8"
    ),
}
CHECKPOINT_SHA256 = "4e2adf98a2fd7ca2ba458c1a99a05973c5744b1db2a0b57e1810af2c56b8be61"


def read_cnn_reference(source, *, smoke=False):
    source = Path(source)
    names = [*REFERENCE_HASHES, "provenance/final.json"]
    if source.is_dir():
        root = source if (source / names[0]).is_file() else source / "milestone2_multi_arrangement"
        raw = {n: (root / n).read_bytes() for n in names}
    else:
        with zipfile.ZipFile(source) as archive:
            raw = {}
            for name in names:
                member = "milestone2_multi_arrangement/" + name
                if archive.namelist().count(member) != 1:
                    raise ValueError("reference requires exactly one copy of each artifact")
                raw[name] = archive.read(member)
    hashes = {n: sha256(raw[n]).hexdigest() for n in REFERENCE_HASHES}
    final = json.loads(raw["provenance/final.json"])
    if any(final.get(n) != h for n, h in hashes.items()):
        raise ValueError("reference artifact hash mismatch")
    if not smoke and hashes != REFERENCE_HASHES:
        raise ValueError("reference identity mismatch")
    manifest = parse_multi_arrangement(raw["data/manifest.json"].decode())
    summary = json.loads(raw["diagnosis/summary.json"])
    if (
        summary["manifest_sha256"] != manifest.sha256
        or summary["checkpoint_sha256"] != final.get("training/last.pt")
        or (
            not smoke
            and (
                summary["checkpoint_sha256"] != CHECKPOINT_SHA256
                or summary["smoke"]
                or summary["completed_steps"] != 8640
            )
        )
    ):
        raise ValueError("reference summary identity mismatch")
    records = [json.loads(line) for line in raw["diagnosis/predictions.jsonl"].splitlines()]
    rows = {
        p.name: [r for r in records if r["arrangement"] == p.name] for p in manifest.populations
    }
    if len(records) != sum(3 * len(p.records) for p in manifest.populations):
        raise ValueError("incomplete reference predictions")
    for p in manifest.populations:
        validate_rows(
            rows[p.name],
            SimpleNamespace(splits={p.name: p.records}),
            p.name,
            summary["arrangements"][p.name]["summary"],
        )
    for role in ("training_fit", "transfer"):
        if aggregate_role(manifest.populations, rows, role) != summary["aggregates"][role]:
            raise ValueError("reference aggregate disagrees with predictions")
    return manifest, summary, rows, hashes
