"""Read and audit the pinned quartet-fit reference without archive extraction."""

import json
import zipfile
from hashlib import sha256
from pathlib import Path

from multimodal_loop.data.quartet_fit import parse_quartet_fit
from multimodal_loop.eval.focused_diagnosis import validate_rows

REFERENCE_HASHES = {
    "training/last.pt": "3e18600b01fcf94514d7a2aed9a1e4bf4d5b21958bd208966e57dee056f1fee4",
    "data/manifest.json": "86bdb774cef97e6b76752f016edd2b73473e8c989b77a7e2cedcb4e295894180",
    "diagnosis/summary.json": "842b57c8edd9606db37fe49d8a8f02b1e6d487d914ffad3566de332e40d5fd9d",
    "diagnosis/training_fit_examples.jsonl": (
        "e9f4898240012e0dd73241ca2b95463a701ef9ac0ead71a240ff709b5e86a41a"
    ),
}


def read_transfer_reference(source, *, smoke=False):
    source = Path(source)
    names = [*REFERENCE_HASHES, "provenance/final.json"]
    if source.is_dir():
        root = (
            source if (source / "training/last.pt").is_file() else source / "milestone2_quartet_fit"
        )
        raw = {name: (root / name).read_bytes() for name in names}
    else:
        with zipfile.ZipFile(source) as archive:
            raw = {}
            for name in names:
                member = "milestone2_quartet_fit/" + name
                if archive.namelist().count(member) != 1:
                    raise ValueError("reference requires exactly one copy of each artifact")
                raw[name] = archive.read(member)
    hashes = {name: sha256(raw[name]).hexdigest() for name in REFERENCE_HASHES}
    final = json.loads(raw["provenance/final.json"])
    if any(final.get(k) != v for k, v in hashes.items()):
        raise ValueError("reference artifact hash mismatch")
    if not smoke and hashes != REFERENCE_HASHES:
        raise ValueError("reference identity mismatch")
    fit = parse_quartet_fit(raw["data/manifest.json"].decode())
    summary = json.loads(raw["diagnosis/summary.json"])
    rows = [json.loads(line) for line in raw["diagnosis/training_fit_examples.jsonl"].splitlines()]
    if summary["checkpoint_sha256"] != hashes["training/last.pt"] or (
        summary["manifest_sha256"] != fit.sha256
    ):
        raise ValueError("reference summary identity mismatch")
    validate_rows(rows, fit, "training_fit", summary["training_fit"])
    return raw, fit, summary, rows, hashes
