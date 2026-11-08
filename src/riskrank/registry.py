"""Model registry (FR5). A published model version is two files in one folder:

    <version>/bundle.joblib   the fitted ModelVersion and its explanation background
    <version>/model.json      metadata, including the SHA-256 of bundle.joblib

In the pipelines the folder is attached to a GitHub release; the settings file names the version
and its expected checksum (FR12). load() refuses a bundle whose checksum differs: NFR3 counts that
as a scoring failure, so the CI hook falls back to the default test order.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import joblib

from riskrank.explain import Background, Explainer, make_background
from riskrank.models import SCALED

FEATURE_PIPELINE_VERSION = "1"


class ChecksumMismatch(RuntimeError):
    pass


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(mv, base_frame, *, version: str, granularity: str, corpus: list[str],
            training_window: dict, training_range: dict | None, root: str, seed: int) -> dict:
    x, _ = mv.pipeline.transform(base_frame, scale=mv.tuned.family in SCALED)
    bg = make_background(x, seed=seed)
    folder = os.path.join(root, version)
    os.makedirs(folder, exist_ok=False)
    bundle = os.path.join(folder, "bundle.joblib")
    joblib.dump({"model_version": mv, "background": bg}, bundle)
    meta = {"version": version, "granularity": granularity, "algorithm": mv.tuned.family,
            "corpus": corpus, "training_window": training_window, "training_range": training_range,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "explanation": Explainer(mv, bg).record(), "model": mv.record(),
            "published": datetime.now(timezone.utc).isoformat(), "bundle_sha256": sha256(bundle)}
    with open(os.path.join(folder, "model.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def load(folder: str, expected_sha256: str):
    """Return (metadata, ModelVersion, Explainer), or raise ChecksumMismatch."""
    bundle = os.path.join(folder, "bundle.joblib")
    actual = sha256(bundle)
    if actual != expected_sha256:
        raise ChecksumMismatch(f"{bundle}: expected {expected_sha256[:12]}, got {actual[:12]}")
    meta = json.load(open(os.path.join(folder, "model.json")))
    if meta["feature_pipeline_version"] != FEATURE_PIPELINE_VERSION:
        raise ChecksumMismatch(f"feature pipeline {meta['feature_pipeline_version']} != {FEATURE_PIPELINE_VERSION}")
    b = joblib.load(bundle)
    bg: Background = b["background"]
    return meta, b["model_version"], Explainer(b["model_version"], bg)
