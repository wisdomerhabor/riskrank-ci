"""Scoring and explanation service (FR6, FR7; NFR3, NFR4, NFR6, NFR14). HTTP via Flask.

    RISKRANK_MODELS=models.json flask --app riskrank.service run      # models.json below

models.json names, per granularity, a registry folder and its expected checksum:
    {"commit": {"folder": "...", "sha256": "..."}, "module": {"folder": "...", "sha256": "..."}}

POST /score
    {"granularity": "commit" | "module", "repository": "owner/name",
     "artefacts": [{"id": "...", "features": {"la": 3, ...}}, ...]}
  -> {"model_version": ..., "transferred": bool, "elapsed_ms": ...,
      "scores": [{"id", "p", "raw", "in_training_range", "reason_codes": [...]}, ...]}
  Only derived metrics and identifiers are accepted; no source code (NFR6).
  422 on a schema mismatch, 503 when the model failed to load (bad checksum) or an explanation is
  missing or non-additive. The CI hook treats both as scoring failure (NFR3).
GET /health -> loaded versions, checksums and input schemas.

The service stores nothing and holds no secret (NFR4). In a CI job it is reachable only on the
job's private network. As a long-lived service on a reachable host, set RISKRANK_TOKEN and every
request must carry "Authorization: Bearer <token>".
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
from flask import Flask, jsonify, request

from riskrank import registry
from riskrank.explain import ExplanationError
from riskrank.features import SchemaMismatch


def create_app(models: dict | None = None, token: str | None = None) -> Flask:
    app = Flask(__name__)
    if models is None:
        with open(os.environ.get("RISKRANK_MODELS", "models.json")) as f:
            models = json.load(f)
    token = token if token is not None else os.environ.get("RISKRANK_TOKEN")

    loaded, failures = {}, {}
    for granularity, spec in models.items():
        try:
            loaded[granularity] = registry.load(spec["folder"], spec["sha256"])
        except (registry.ChecksumMismatch, FileNotFoundError) as e:
            failures[granularity] = str(e)

    @app.before_request
    def authenticate():
        if token and request.headers.get("Authorization") != f"Bearer {token}":
            return jsonify(error="unauthenticated"), 401

    @app.get("/health")
    def health():
        return jsonify(
            models={g: {"version": m["version"], "sha256": m["bundle_sha256"],
                        "features": m["model"]["schema"]["features_in"]} for g, (m, _, _) in loaded.items()},
            failures=failures)

    @app.post("/score")
    def score():
        t0 = time.perf_counter()
        body = request.get_json(force=True)
        g = body.get("granularity")
        if g in failures:
            return jsonify(error="model unavailable", detail=failures[g]), 503
        if g not in loaded:
            return jsonify(error=f"unknown granularity {g!r}"), 422
        meta, mv, explainer = loaded[g]
        expected = mv.pipeline.features
        arts = body.get("artefacts", [])
        for a in arts:
            if set(a.get("features", {})) != set(expected):
                return jsonify(error="schema mismatch", expected=expected,
                               got=sorted(a.get("features", {}))), 422
        frame = pd.DataFrame([a["features"] for a in arts], columns=expected)
        try:
            p, raw = mv.score(frame)
            codes = explainer.reason_codes(frame)
        except (SchemaMismatch, ValueError) as e:
            return jsonify(error="schema mismatch", detail=str(e)), 422
        except ExplanationError as e:
            return jsonify(error="explanation failed", detail=str(e)), 503
        rng = meta.get("training_range")
        out = []
        for i, a in enumerate(arts):
            f = a["features"]
            inside = True
            if rng:
                inside = f.get("nf", 0) <= rng["max_files"] and f.get("la", 0) + f.get("ld", 0) <= rng["max_lines"]
            out.append({"id": a["id"], "p": float(p[i]), "raw": float(raw[i]),
                        "in_training_range": bool(inside), "reason_codes": codes[i]})
        return jsonify(model_version=meta["version"],
                       transferred=body.get("repository") not in meta["corpus"],
                       elapsed_ms=round((time.perf_counter() - t0) * 1000, 2), scores=out)

    return app
