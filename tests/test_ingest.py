import copy
import io
import json
import zipfile
from datetime import datetime, timezone

import urllib.error

from riskrank.ingest import check
from riskrank.store import Store
from tests.test_store import PLAN


def zipped(plan, durations=None):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("plan.json", json.dumps(plan))
        if durations is not None:
            z.writestr("durations.json", json.dumps(durations))
    return b.getvalue()


def run(i, **kw):
    r = {"id": i, "path": ".github/workflows/riskrank.yml", "event": "push", "head_branch": "master",
         "head_sha": "a1", "created_at": "2026-11-08T10:00:00Z"}
    return r | kw


class FakeGitHub:
    def __init__(self, runs, artifacts, blobs):
        self.runs, self.artifacts, self.blobs, self.calls = runs, artifacts, blobs, 0

    def json(self, path):
        if path == "/repos/o/r":
            return {"default_branch": "master"}
        if "/workflows/" in path:
            return {"workflow_runs": self.runs}
        if path.startswith("/repos/o/r/actions/runs/"):
            return {"artifacts": self.artifacts.get(int(path.split("/")[-2]), [])}
        if "/releases" in path:
            return []
        raise AssertionError(path)

    def download(self, url):
        self.calls += 1
        blob = self.blobs[url]
        if isinstance(blob, Exception):
            raise blob
        return blob


def art(url, expired=False):
    return [{"name": "riskrank-plan", "archive_download_url": url, "expired": expired}]


def test_each_outcome_is_recorded_once(tmp_path):
    other_sha = copy.deepcopy(PLAN)
    bad_format = copy.deepcopy(PLAN)
    bad_format["format"] = 99
    gh = FakeGitHub(
        runs=[run(1), run(2, head_branch="feature"), run(3, path=".github/workflows/other.yml"), run(4), run(5),
              run(6), run(7, head_sha="zz"), run(8), run(9, event="pull_request")],
        artifacts={1: art("u1"), 4: art("u4", expired=True), 5: art("u5"), 7: art("u7"), 8: art("u8")},
        blobs={"u1": zipped(PLAN, {"t::a": 0.25}), "u5": urllib.error.URLError("down"), "u7": zipped(other_sha),
               "u8": zipped(bad_format)})
    store = Store(f"sqlite:///{tmp_path}/t.db")
    now = datetime(2026, 11, 9, tzinfo=timezone.utc)
    counts = check(gh, store, "o/r", now=now)
    assert counts == {"stored": 1, "rejected": 5, "lost": 1, "retry": 1, "missing": 1}
    log = store.ingest_outcomes("o/r").set_index("build_ref")["outcome"].to_dict()
    assert log == {"1": "stored", "2": "rejected", "3": "rejected", "4": "lost", "5": "retry", "6": "missing",
                   "7": "rejected", "8": "rejected", "9": "rejected"}
    rid = store.runtime_runs("o/r")["id"].iloc[0]
    assert store.test_plan(int(rid)).loc[0, "duration"] == 0.25

    # the next check revisits only the run to retry; past the retention period it is recorded as lost
    gh.calls = 0
    counts = check(gh, store, "o/r", now=datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert counts == {"lost": 1} and gh.calls == 1
    assert check(gh, store, "o/r", now=now) == {}
