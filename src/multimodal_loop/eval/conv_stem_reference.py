"""Pinned transformer and CNN comparisons on identical arrangement populations."""

from multimodal_loop.eval.cnn_reference import read_arrangement_reference, read_cnn_reference
from multimodal_loop.train.conv_stem_checkpoint import MANIFEST_SHA256

CNN_HASHES = {
    "data/manifest.json": MANIFEST_SHA256,
    "diagnosis/summary.json": "11939f4e4b671efe20746a3b94c11395992064f39b41d7e77cebc26767becd54",
    "diagnosis/predictions.jsonl": (
        "0868e98a14d54af9997c29cc824b83adbd4b356cf6f4e77de9cadb237bf48019"
    ),
}
CNN_CHECKPOINT_SHA256 = "9e47a412cef98c24d7d5ea03a028e3c499d9d20340db0900876137232b82637d"


def read_conv_stem_references(transformer_source, cnn_source, *, smoke=False):
    transformer = read_cnn_reference(transformer_source, smoke=smoke)
    cnn = read_arrangement_reference(
        cnn_source,
        prefix="milestone2_cnn_baseline",
        expected_hashes=CNN_HASHES,
        checkpoint_sha256=CNN_CHECKPOINT_SHA256,
        smoke=smoke,
    )
    if transformer[0].sha256 != cnn[0].sha256:
        raise ValueError("reference populations disagree")
    return {"transformer": transformer, "cnn": cnn}
