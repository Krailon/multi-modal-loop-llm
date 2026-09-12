"""Audit saved single-arrangement transfer predictions without loading its model."""

import json
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from multimodal_loop.data.quartet_transfer import build_transfer_populations, transfer_manifest
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.quartet_transfer_reference import REFERENCE_HASHES

TRANSFER_HASHES = {
    "diagnosis/summary.json": "2d807409f12e378aff116ffb1a92c4df19de2cd86cd26783a02874e6c7a2d835",
    "diagnosis/predictions.jsonl": (
        "17ff66263223046dba7f2e21030079d3583361873e86d3decbd236096d8ad1a8"
    ),
    "data/transfer_manifest.json": (
        "bc88c065e17e7ba449c7d209bd9816f1d5384c106f51117a8159f3eb1d70eb03"
    ),
}


def read_multi_reference(source, fit, *, smoke=False):
    source = Path(source)
    names = [*TRANSFER_HASHES, "provenance/final.json"]
    if source.is_dir():
        root = (
            source
            if (source / "diagnosis/predictions.jsonl").is_file()
            else (source / "milestone2_quartet_transfer")
        )
        raw = {name: (root / name).read_bytes() for name in names}
    else:
        with zipfile.ZipFile(source) as archive:
            raw = {}
            for name in names:
                member = "milestone2_quartet_transfer/" + name
                if archive.namelist().count(member) != 1:
                    raise ValueError("reference requires exactly one copy of each artifact")
                raw[name] = archive.read(member)
    hashes = {name: sha256(raw[name]).hexdigest() for name in TRANSFER_HASHES}
    final = json.loads(raw["provenance/final.json"])
    if any(final.get(k) != v for k, v in hashes.items()):
        raise ValueError("transfer reference artifact hash mismatch")
    if not smoke and hashes != TRANSFER_HASHES:
        raise ValueError("transfer reference identity mismatch")
    populations = build_transfer_populations(fit)
    content, _ = transfer_manifest(fit, populations)
    if raw["data/transfer_manifest.json"].decode() != content:
        raise ValueError("reference population differs from source fit corpus")
    summary = json.loads(raw["diagnosis/summary.json"])
    if summary["fit_manifest_sha256"] != fit.sha256 or not summary["reproduction"]["passed"]:
        raise ValueError("reference requires matching fit identity and successful reproduction")
    if not smoke and summary["checkpoint_sha256"] != REFERENCE_HASHES["training/last.pt"]:
        raise ValueError("reference checkpoint identity mismatch")
    records = [json.loads(line) for line in raw["diagnosis/predictions.jsonl"].splitlines()]
    if len(records) != 1728 * len(populations):
        raise ValueError("incomplete reference predictions")
    rows = {p.name: [r for r in records if r["arrangement"] == p.name] for p in populations}
    for p in populations:
        validate_rows(
            rows[p.name],
            SimpleNamespace(splits={p.name: p.records}),
            p.name,
            summary["arrangements"][p.name]["summary"],
        )
    return summary, rows, hashes
