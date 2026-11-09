import json
import os

import numpy as np
import pandas as pd
import pytest

from riskrank import benchmarks, models, run_benchmarks as rb


def _frame(n, cols, prefix, seed):
    """Unit-test fixture only. Never used as evidence (section 3.5.1)."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({c: rng.gamma(2.0, 10.0, n) for c in cols})
    lead = x[cols[0]]
    y = rng.random(n) < 1 / (1 + np.exp(-(lead - lead.mean()) / lead.std()))
    x["label"] = y
    x["size"] = x[cols[0]].round() + 1
    x["id"] = [f"{prefix}:{i}" for i in range(n)]
    return x


def synthetic():
    nasa_cols = ["LOC_TOTAL", "A", "B"]
    nasa = {name: _frame(160, nasa_cols + ([f"X{name}"] if name == "KC3" else []), name, i)
            for i, name in enumerate(["KC3", "MC2", "CM1"])}
    ck = list(benchmarks.CK)
    ck = ["loc"] + [c for c in ck if c != "loc"]
    promise = {s: [(f"{s}-{v}", _frame(140, ck, f"{s}-{v}", 10 * j + k)[list(benchmarks.CK) + rb.META])
                   for k, v in enumerate(["1.0", "1.1", "1.2"])] for j, s in enumerate(["ivy", "log4j"])}
    return {"nasa": nasa, "promise": promise}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "N_JOBS", 1)
    monkeypatch.setattr(rb, "CONFIG", "config/benchmarks.yaml")
    rb._init(synthetic)
    yield str(tmp_path / "bench")
    rb.DATA.clear()


def test_score_refuses_before_selection(run):
    rb.main(["tune", "--out", run, "--check", "--jobs", "1", "--families", "lr", "--corpora", "promise",
             "--modes", "within"])
    with pytest.raises(SystemExit, match="select phase"):
        rb.main(["score", "--out", run, "--jobs", "1"])


def test_pipeline_check_end_to_end(run):
    rb.main(["all", "--out", run, "--check", "--jobs", "1", "--families", "lr", "rf"])
    m = json.load(open(os.path.join(run, "manifest.json")))
    assert m["pipeline_check"] and m["settings"]["trials"] == 1
    tuned = rb._read_jsonl(os.path.join(run, "tuned.jsonl"))
    # nasa within: 2 projects x 3 rounds; promise within: 2 systems x 2 pairs; cross: 2 + 2; all x 2 families
    assert len(tuned) == (6 + 4 + 2 + 2) * 2
    assert not [t for t in tuned if "error" in t]
    rep = json.load(open(os.path.join(run, "report.json")))
    for corpus in ("nasa", "promise"):
        within = rep[f"{corpus}_within"]
        assert within["family"] in ("lr", "rf") and len(within["family_tests"]) == 8
        assert "recency_lines" not in next(iter(within["per_project"].values()))[within["family"]]["recall_batch"]
    # cross-project NASA uses only the shared metrics: KC3's extra metric never reaches a model
    t = {"corpus": "nasa", "mode": "cross", "project": "MC2", "unit": "holdout", "family": "lr"}
    assert rb.split(t, m["settings"])[3] == ["LOC_TOTAL", "A", "B"]


def test_refit_in_score_phase_reproduces_tuning(run):
    s = rb._settings(True, ["promise"], ["within"], ["rf"])
    t = {"corpus": "promise", "mode": "within", "project": "ivy", "unit": "ivy-1.0>ivy-1.1", "family": "rf"}
    a, b = rb.tune_task(t, s), rb.tune_task(t, s)
    assert (a["params"], a["cv_aucpr"]) == (b["params"], b["cv_aucpr"])
    r1, _ = rb.score_task(a, s)
    r2, _ = rb.score_task(a, s)
    assert r1[0]["aucpr"] == r2[0]["aucpr"]


@pytest.mark.skipif(not os.path.isdir(benchmarks.PROMISE_DIR), reason="benchmark data not downloaded")
def test_every_module_id_is_unique_within_its_release():
    # The PROMISE files repeat the header "name" (system, then class); reading the first gave every class
    # in a release the same id, which broke tie-breaks and collapsed the pooled predictions.
    for system, releases in benchmarks.promise().items():
        for release, x in releases:
            assert not x["id"].duplicated().any(), release
    for name, x in benchmarks.nasa().items():
        assert not x["id"].duplicated().any(), name
