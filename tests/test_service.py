"""Service tests. The models here are trained on a small synthetic fixture: unit tests only, never evidence."""

import json

import numpy as np
import pytest

from riskrank import registry
from riskrank.explain import Explainer, make_background
from riskrank.models import SCALED, fit_final, tune
from riskrank.service import create_app
from tests.test_models import FEATS, synthetic_commits

RANGE = {"max_files": 100, "max_lines": 10000}


@pytest.fixture(scope="module", params=["lr", "rf", "lgbm", "mlp"])
def published(request, tmp_path_factory):
    d = synthetic_commits(700)
    base, calib = d.iloc[:500], d.iloc[500:]
    mv = fit_final(base, calib, FEATS, tune(base, FEATS, request.param, 0.3, 1), seed=1)
    root = tmp_path_factory.mktemp("reg")
    meta = registry.publish(mv, base, version=f"jit-{request.param}", granularity="commit",
                            corpus=["apache/x"], training_window={}, training_range=RANGE,
                            root=str(root), seed=1)
    return str(root / f"jit-{request.param}"), meta, mv, base


def client(published, **kw):
    folder, meta, _, _ = published
    return create_app({"commit": {"folder": folder, "sha256": meta["bundle_sha256"]}}, **kw).test_client()


def body(features, repo="pallets/flask"):
    return {"granularity": "commit", "repository": repo,
            "artefacts": [{"id": f"c{i}", "features": f} for i, f in enumerate(features)]}


def test_scores_have_reason_codes_and_are_additive(published):
    _, _, mv, base = published
    feats = base[FEATS].head(5).to_dict("records")
    r = client(published).post("/score", json=body(feats))
    assert r.status_code == 200
    out = r.get_json()
    assert out["transferred"] is True
    for s in out["scores"]:
        assert 0 <= s["p"] <= 1 and len(s["reason_codes"]) == len(mv.pipeline.kept_)
        contribs = [abs(c["contribution"]) for c in s["reason_codes"]]
        assert contribs == sorted(contribs, reverse=True)
    x, _ = mv.pipeline.transform(base.head(5), scale=mv.tuned.family in SCALED)
    ex = Explainer(mv, make_background(mv.pipeline.transform(base, scale=mv.tuned.family in SCALED)[0], 1))
    b, c = ex.contributions(x)  # raises if additivity fails
    assert np.allclose(b + c.sum(1), [s["raw"] for s in out["scores"]], atol=1e-6)


def test_same_request_gives_identical_reason_codes(published):
    _, _, _, base = published
    feats = base[FEATS].head(3).to_dict("records")
    c = client(published)
    a, b = c.post("/score", json=body(feats)).get_json(), c.post("/score", json=body(feats)).get_json()
    assert [s["reason_codes"] for s in a["scores"]] == [s["reason_codes"] for s in b["scores"]]


def test_training_range_flag(published):
    r = client(published).post("/score", json=body([{"la": 9000, "ld": 2000, "nf": 3}])).get_json()
    assert r["scores"][0]["in_training_range"] is False


def test_schema_mismatch_is_rejected(published):
    r = client(published).post("/score", json=body([{"la": 1, "ld": 1}]))
    assert r.status_code == 422


def test_wrong_checksum_means_model_unavailable(published):
    folder, _, _, _ = published
    app = create_app({"commit": {"folder": folder, "sha256": "0" * 64}})
    r = app.test_client().post("/score", json=body([{"la": 1, "ld": 1, "nf": 1}]))
    assert r.status_code == 503
    assert "commit" in app.test_client().get("/health").get_json()["failures"]


def test_token_required_when_configured(published):
    c = client(published, token="s3cret")
    assert c.get("/health").status_code == 401
    assert c.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_background_never_stores_a_record_and_merges_small_clusters():
    import pandas as pd
    rng = np.random.default_rng(0)
    x = pd.DataFrame(np.vstack([rng.normal(0, 1, (200, 2)), [[50, 50], [51, 51]]]), columns=["a", "b"])
    bg = make_background(x, seed=0, k=6)
    assert bg.merged >= 1  # the two-point outlier cluster cannot stand alone
    assert not any(((bg.centres.values == row).all(1)).any() for row in x.values)
    assert bg.weights.sum() == pytest.approx(1.0)


def test_service_starts_before_models_json_exists(published, tmp_path, monkeypatch):
    # In a CI job the service container starts before the step that writes models.json.
    folder, meta, _, _ = published
    path = tmp_path / "models.json"
    monkeypatch.setenv("RISKRANK_MODELS", str(path))
    c = create_app().test_client()  # must not raise
    assert c.post("/score", json=body([{"la": 1, "ld": 1, "nf": 1}])).status_code == 503
    path.write_text(json.dumps({"commit": {"folder": folder, "sha256": meta["bundle_sha256"]}}))
    assert c.post("/score", json=body([{"la": 1, "ld": 1, "nf": 1}])).status_code == 200


def test_neural_reason_codes_are_approximate_but_repeatable(published):
    folder, meta, mv, base = published
    if mv.tuned.family != "mlp":
        pytest.skip("neural model only")
    from riskrank.explain import Explainer, make_background
    x, _ = mv.pipeline.transform(base, scale=True)
    ex = Explainer(mv, make_background(x, seed=1))
    rows = base.iloc[:3]
    assert ex.reason_codes(rows) == ex.reason_codes(rows)            # same input, same codes
    assert ex.record()["attribution"] == "Kernel SHAP (approximate)"
