"""The CI hook (FR8, FR9, FR14, FR16-FR18; NFR3, NFR6, NFR19). The thin, CI-specific part is the
workflow file; everything here is platform-neutral (NFR14).

    python -m riskrank.hook --repo . --before $BEFORE --after $SHA --settings riskrank.yml \
        --service http://scoring:8000 --coverage coverage-map.json --tests tests.txt --out plan

Writes into --out:
    plan.json       the full plan (ranking, scores, reason codes, ordered tests, counts, settings used)
    order.txt       test ids in plan order, for the test step
    in_budget.txt   the in-budget test ids
    summary.md      the build annotation (risk summary + reference to the plan)
It never fails the build. Any failure leads to a warning and the default test order (NFR3).
NFR19: author identifiers stay inside this process; none is written to any output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

import pandas as pd

from riskrank import settings as settings_mod
from riskrank.budget import mark, rank, realised_share
from riskrank.jit_features import FEATURES as JIT_FEATURES
from riskrank.jit_features import History
from riskrank.mining.snapshots import history as module_history
from riskrank.mining.snapshots import process_window, static_metrics
from riskrank.plan_format import PLAN_FORMAT
from riskrank.testplan import build_plan

MODULE_FEATURES = ["size", "complexity", "churn", "changes", "prior_fixes"]
FIX_KEYWORDS = r"\b(fix|fixes|fixed|fixing|bug|bugs|defect|defects|patch|patched|patches)\b"


def read_tests(path: str | None) -> list[str]:
    """Current test ids, as JSON lines written by riskrank.order_plugin (RISKRANK_COLLECT)."""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def is_source(path: str, roots: list[str], exclude: list[str]) -> bool:
    return path.endswith(".py") and any(path.startswith(r) for r in roots) and not any(path.startswith(e) for e in exclude)


def touched_modules(repo: str, before: str, after: str, roots, exclude) -> list[str]:
    names = git(repo, "diff", "--name-only", "--diff-filter=d", f"{before}..{after}").split()
    return sorted({p for p in names if is_source(p, roots, exclude)})


def module_features(repo: str, after: str, modules: list[str], months: int, keywords: str) -> pd.DataFrame:
    when = pd.Timestamp(git(repo, "show", "-s", "--format=%cI", after).strip()).tz_convert("UTC")
    commits = module_history(after, repo=repo)
    proc = process_window(commits, modules, when - pd.DateOffset(months=months), when + pd.Timedelta(seconds=1),
                          re.compile(keywords, re.IGNORECASE))
    rows = []
    for m in modules:
        sm = static_metrics(after, m, repo=repo) or (None, None)
        rows.append({"id": m, "size": sm[0], "complexity": sm[1], **proc[m]})
    return pd.DataFrame(rows, columns=["id", *MODULE_FEATURES])  # a push may touch no module at all


def default_order(out: str, tests: list[str], reason: str, settings_used: dict) -> dict:
    plan = {"format": PLAN_FORMAT, "status": "default order", "reason": reason, "settings": settings_used,
            "tests": [{"test": t, "position": i + 1, "in_budget": False} for i, t in enumerate(tests)]}
    _write(out, plan, tests, [], f"## RiskRank\n\n⚠️ **Default test order.** {reason}\n")
    print(f"::warning title=RiskRank::{reason} Running the default test order.")
    return plan


def commit_level(a, s: dict, shas: list[str], team: set[str], timeout: float) -> dict:
    """Score the deploying team's commits with the just-in-time model (FR8). Without a commit model in
    the settings, escalation is not configured. A scoring failure leaves the run partial (NFR3)."""
    if not (s.get("models") or {}).get("commit"):
        return {"status": "not configured", "commits": []}
    try:
        h = History(a.repo, a.after)
        team_shas = [x for x in shas if x in h.by_sha and h.by_sha[x].author in team]
        if not team_shas:
            return {"status": "no team commits", "commits": []}
        # send exactly the features the deployed model expects (the commit model drops asexp, A4)
        with urllib.request.urlopen(f"{a.service}/health", timeout=timeout) as hr:
            expected = json.loads(hr.read())["models"]["commit"]["features"]
        r = post(f"{a.service}/score", {"granularity": "commit", "repository": s.get("repository"),
                 "artefacts": [{"id": x, "features": {f: h.features(x)[f] for f in expected}}
                               for x in team_shas]}, timeout)
        return {"status": "assessed", "model_version": r["model_version"], "transferred": r["transferred"],
                "commits": [{"id": c["id"], "p": c["p"], "in_training_range": c["in_training_range"],
                             "above_threshold": c["p"] > s["risk_threshold"],
                             "reason_codes": c["reason_codes"][:5]} for c in r["scores"]]}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
        return {"status": "not assessed", "error": f"{type(e).__name__}", "commits": []}


def run(a) -> dict:
    t_start = time.perf_counter()
    tests = read_tests(a.tests)
    raw, problems = settings_mod.load(a.settings)
    if problems:
        return default_order(a.out, tests, "Invalid settings: " + "; ".join(problems), raw)
    s = raw
    roots, exclude = s.get("source_roots", ["src/"]), s.get("exclude_paths", ["tests/"])
    timeout = float(s["scoring_timeout_seconds"])
    team = {e.lower() for e in s.get("deploying_team", [])}

    # ---- module level: the ranking and the test plan
    modules = touched_modules(a.repo, a.before, a.after, roots, exclude)
    try:
        mf = module_features(a.repo, a.after, modules, s.get("process_period_months", 12), s.get("fix_keywords", FIX_KEYWORDS))
        unparsable = mf[mf["size"].isna()]["id"].tolist()
        mf = mf.dropna(subset=["size"])
        t0 = time.perf_counter()
        resp = post(f"{a.service}/score", {"granularity": "module", "repository": s.get("repository"),
                    "artefacts": [{"id": r.id, "features": {f: float(getattr(r, f)) for f in MODULE_FEATURES}}
                                  for r in mf.itertuples()]}, timeout) if len(mf) else {"scores": [], "model_version": None}
        module_seconds = time.perf_counter() - t0
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, subprocess.CalledProcessError) as e:
        return default_order(a.out, tests, f"Module scoring failed ({type(e).__name__}: {e}).", s)

    scored = pd.DataFrame(resp["scores"])
    if len(scored):
        scored = scored.merge(mf[["id", "size"]], on="id")
        ranked = rank(scored, s["ranking_key"])
        ranked["in_budget"] = mark(ranked, s["budget_unit"], s["budget"] / 100, s["ranking_key"]).to_numpy()
        ranked["flagged"] = ranked["p"] > s["risk_threshold"]
    else:
        ranked = pd.DataFrame(columns=["id", "p", "raw", "size", "in_budget", "flagged", "reason_codes"])
    zero = set(ranked.loc[ranked["size"] <= 0, "id"]) if len(ranked) else set()
    ranked_pairs = [(r.id, bool(r.in_budget)) for r in ranked.itertuples() if r.id not in zero]

    coverage, map_meta = {}, {}
    if a.coverage and os.path.exists(a.coverage):
        cm = json.load(open(a.coverage))
        coverage, map_meta = {t: set(m) for t, m in cm["tests"].items()}, cm.get("meta", {})
    else:
        return default_order(a.out, tests, "Coverage map unavailable.", s)
    since = committed = None
    if map_meta.get("commit"):
        try:
            since = int(git(a.repo, "rev-list", "--count", f"{map_meta['commit']}..{a.after}").strip())
        except subprocess.CalledProcessError:
            since = None
        try:  # the map's own date, so "N commits ago" can carry a time (dashboard)
            committed = git(a.repo, "show", "-s", "--format=%cI", map_meta["commit"]).strip() or None
        except subprocess.CalledProcessError:
            committed = None
    plan_t0 = time.perf_counter()
    tp = build_plan(ranked_pairs, zero, coverage, tests, durations=map_meta.get("durations", {}),
                    map_commit=map_meta.get("commit"), commits_since_map=since,
                    map_age_intervals=map_meta.get("age_intervals"), map_committed_at=committed,
                    map_collected_at=map_meta.get("collected_at"))
    plan_seconds = time.perf_counter() - plan_t0

    # ---- commit level: the risk summary and build escalation (FR8, FR16)
    shas = git(a.repo, "rev-list", "--no-merges", f"{a.before}..{a.after}").split()
    jit = commit_level(a, s, shas, team, timeout)
    escalate = any(c["above_threshold"] and c["in_training_range"] for c in jit["commits"])

    plan = {
        "format": PLAN_FORMAT,
        "status": "partial" if jit["status"] == "not assessed" else "complete",  # partial: module plan kept
        "settings": {k: v for k, v in s.items() if k != "deploying_team"},
        "push": {"before": a.before, "after": a.after, "commits": len(shas)},
        "module_model": resp.get("model_version"), "module_transferred": resp.get("transferred"),
        "modules_touched": len(modules),
        "modules_unparsable": unparsable,
        "ranking": [{"rank": i + 1, "module": r.id, "p": r.p, "raw": r.raw, "size": r.size, "in_budget": bool(r.in_budget),
                     "flagged": bool(r.flagged), "reason_codes": r.reason_codes}
                    for i, r in enumerate(ranked.itertuples())],
        "realised_share": realised_share(ranked, ranked["in_budget"].astype(bool), s["budget_unit"]) if len(ranked) else None,
        "test_plan": tp.to_json(), "commit_level": jit, "escalated": escalate,
        "timing_seconds": {"module_scoring": round(module_seconds, 3), "test_ordering": round(plan_seconds, 3),
                           "hook_total": round(time.perf_counter() - t_start, 3)},
    }
    order = [i.test for i in tp.items]
    _write(a.out, plan, order, [i.test for i in tp.items if i.in_budget], summary(plan))
    if escalate:
        print("::warning title=RiskRank escalation::A commit's risk exceeds the threshold. See the plan.")
    return plan


def _noun(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def summary(plan: dict) -> str:
    st = plan["test_plan"]["stats"]
    flagged = [r for r in plan["ranking"] if r["flagged"]]
    jit = plan["commit_level"]
    lines = ["## RiskRank risk summary", "",
             f"**{plan['modules_touched']}** {_noun(plan['modules_touched'], 'module')} touched, **{len(plan['ranking'])}** ranked; "
             f"**{st['in_budget_count']}** of {st['tests_total']} tests in budget "
             f"({st['in_budget_fraction']:.0%} of the suite).", ""]
    if flagged:
        lines += ["**Flagged modules** (risk above threshold):", ""] + [f"- module `{r['module']}`: {r['p']:.2f}" for r in flagged] + [""]
    above = [c for c in jit["commits"] if c["above_threshold"]]
    outside = [c for c in jit["commits"] if not c["in_training_range"]]
    if jit["status"] == "not configured":
        lines += ["Build escalation: **not configured** (no just-in-time model in the settings).", ""]
    else:
        lines += [f"Commit-level risk: {jit['status']}; {len(above)} {_noun(len(above), 'commit')} above threshold"
                  + (f", {len(outside)} outside the training range" if outside else "") + ".",
                  f"Build escalated: **{'yes' if plan['escalated'] else 'no'}** (escalation never fails the build).", ""]
    lines += ["Full plan: the `riskrank-plan` artifact of this run."]
    if st["stale_map_warning"]:
        lines.insert(2, "⚠️ The coverage map is more than two refresh intervals old.")
    return "\n".join(lines) + "\n"


def _write(out: str, plan: dict, order: list[str], in_budget: list[str], md: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1, default=str)
    open(os.path.join(out, "order.txt"), "w").write("\n".join(order) + ("\n" if order else ""))
    open(os.path.join(out, "in_budget.txt"), "w").write("\n".join(in_budget) + ("\n" if in_budget else ""))
    open(os.path.join(out, "summary.md"), "w").write(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("repo", "before", "after", "settings", "service", "coverage", "tests", "out"):
        ap.add_argument(f"--{k}", required=k in ("before", "after"))
    ap.set_defaults(repo=".", settings="riskrank.yml", service="http://localhost:8000", out="riskrank-plan")
    a = ap.parse_args()
    try:
        run(a)
    except Exception as e:  # NFR3: the hook never fails the build
        tests = read_tests(a.tests)
        default_order(a.out, tests, f"Unexpected error ({type(e).__name__}: {e}).", {})


if __name__ == "__main__":
    main()
