"""The plan-ingest component (FR10; NFR5, NFR7; Table 3.8). Runs on the workstation and pulls, because a
hosted runner cannot reach it.

    python -m riskrank.ingest --repo wisdomerhabor/riskrank-ci --db sqlite:///riskrank.db
    python -m riskrank.ingest --repo wisdomerhabor/flask --db sqlite:///riskrank.db --every 600

At each check it lists completed runs of the scoring workflow and, for each run not yet stored:

* accepts a plan file only from a push run of the specified workflow file on the repository's default
  branch, whose plan describes that run's own commit, and only if it matches the plan-file format;
  anything else is rejected and the rejection recorded;
* records a completed run with no plan artefact as missing, and one whose artefact has expired (7-day
  retention) as lost; a failed download is retried at the next check;
* stores the plan and the test-run report (durations.json) through the storage interface, and copies
  the metadata of each model version the plan names from its release asset, once.

The token is read from RISKRANK_GH_TOKEN (or the file named by RISKRANK_GH_TOKEN_FILE), kept outside the
repository; it needs read access to Actions artefacts and releases only. Every request uses HTTPS.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

from riskrank.plan_format import problems
from riskrank.store import Store

API = "https://api.github.com"
WORKFLOW = ".github/workflows/riskrank.yml"
ARTIFACT = "riskrank-plan"
REGISTRY = "wisdomerhabor/riskrank-ci"
RETENTION = timedelta(days=7)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class GitHub:
    """Minimal read-only GitHub REST client. Artefact downloads redirect to blob storage; the token is
    sent to api.github.com only, never to the redirect target."""

    def __init__(self, token: str | None):
        self.token = token
        self._noredirect = urllib.request.build_opener(_NoRedirect)

    def _req(self, url: str) -> urllib.request.Request:
        assert url.startswith("https://"), "every transfer uses HTTPS (NFR5)"
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
             "User-Agent": "riskrank-ingest"}
        if self.token and url.startswith(API):
            h["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(url, headers=h)

    def json(self, path: str) -> dict:
        with urllib.request.urlopen(self._req(API + path), timeout=60) as r:
            return json.loads(r.read())

    def download(self, url: str) -> bytes:
        try:
            with self._noredirect.open(self._req(url), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                target = e.headers["Location"]
                assert target.startswith("https://")
                with urllib.request.urlopen(urllib.request.Request(target), timeout=120) as r:
                    return r.read()
            raise


def read_token() -> str | None:
    if os.environ.get("RISKRANK_GH_TOKEN"):
        return os.environ["RISKRANK_GH_TOKEN"]
    f = os.environ.get("RISKRANK_GH_TOKEN_FILE")
    return open(f).read().strip() if f and os.path.exists(f) else None


def accept(run: dict, default_branch: str, workflow: str = WORKFLOW) -> str | None:
    """Why a run's plan must be rejected, or None when it comes from the right workflow and branch."""
    if run.get("path") != workflow:
        return f"run is from {run.get('path')!r}, not the scoring workflow {workflow}"
    if run.get("event") != "push":
        return f"run was triggered by {run.get('event')!r}, not a push"
    if run.get("head_branch") != default_branch:
        return f"run is on {run.get('head_branch')!r}, not the default branch {default_branch!r}"
    return None


def plan_files(blob: bytes) -> tuple[dict, dict]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        plan = json.loads(z.read("plan.json"))
        durations = json.loads(z.read("durations.json")) if "durations.json" in z.namelist() else {}
    return plan, durations


def check(gh: GitHub, store: Store, repo: str, now: datetime | None = None, per_page: int = 100) -> dict:
    """One check: returns counts per outcome for this pass."""
    now = now or datetime.now(timezone.utc)
    default = gh.json(f"/repos/{repo}")["default_branch"]
    runs = gh.json(f"/repos/{repo}/actions/workflows/{os.path.basename(WORKFLOW)}/runs"
                   f"?status=completed&per_page={per_page}")["workflow_runs"]
    counts: dict[str, int] = {}
    seen = store.ingest_outcomes(repo)
    handled = set(seen.loc[seen["outcome"] != "retry", "build_ref"])

    def log(ref: str, outcome: str, reason: str | None = None) -> None:
        store.log_ingest(repo, ref, outcome, reason)
        counts[outcome] = counts.get(outcome, 0) + 1

    for run in runs:
        ref = str(run["id"])
        if ref in handled or store.has_run(repo, ref):
            continue
        why = accept(run, default)
        if why:
            log(ref, "rejected", why)
            continue
        arts = [x for x in gh.json(f"/repos/{repo}/actions/runs/{ref}/artifacts")["artifacts"] if x["name"] == ARTIFACT]
        created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        if not arts:
            log(ref, "missing", "completed build with no plan file")
            continue
        if arts[0].get("expired"):
            log(ref, "lost", "plan file expired before collection")
            continue
        try:
            plan, durations = plan_files(gh.download(arts[0]["archive_download_url"]))
        except (urllib.error.URLError, OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
            if now - created > RETENTION:
                log(ref, "lost", f"not collected within the retention period ({type(e).__name__})")
            else:
                log(ref, "retry", f"{type(e).__name__}: {e}")
            continue
        bad = problems(plan)
        if not bad and plan.get("push", {}).get("after") not in (None, run["head_sha"]):
            bad = [f"plan describes {plan['push']['after']}, the run built {run['head_sha']}"]
        if bad:
            log(ref, "rejected", "plan-file format: " + "; ".join(bad))
            continue
        for g, m in (plan.get("settings", {}).get("models") or {}).items():
            if m and not store.model(m["version"]):
                rec = model_record(gh, m["version"])
                if rec:
                    store.put_model(rec)
        store.put_runtime_run(repo, ref, run["created_at"], plan, durations)
        log(ref, "stored")
    return counts


def model_record(gh: GitHub, version: str, registry: str = REGISTRY) -> dict | None:
    """The model metadata (model.json) from the release asset that holds a version."""
    for rel in gh.json(f"/repos/{registry}/releases?per_page=100"):
        for asset in rel.get("assets", []):
            if asset["name"] == f"models-{version}.tar.gz":
                blob = gh.download(asset["browser_download_url"])
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
                    return json.load(t.extractfile(f"{version}/model.json"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True, help="owner/name; repeat for each deployment")
    ap.add_argument("--db", default=os.environ.get("RISKRANK_DB", "sqlite:///riskrank.db"))
    ap.add_argument("--every", type=float, default=0, help="seconds between checks; 0 = check once")
    a = ap.parse_args()
    gh, store = GitHub(read_token()), Store(a.db)
    while True:
        for repo in a.repo:
            try:
                print(repo, json.dumps(check(gh, store, repo)), flush=True)
            except (urllib.error.URLError, OSError) as e:  # a failed check is retried at the next one
                print(repo, f"check failed: {type(e).__name__}: {e}", flush=True)
        if not a.every:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
