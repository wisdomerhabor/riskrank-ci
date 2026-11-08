"""The repository settings file, riskrank.yml (FR12), and its validation (section 3.3.2).

    budget: 20                     # percent, > 0 and <= 100
    budget_unit: lines             # artefacts | lines
    ranking_key: default           # default | effort
    risk_threshold: 0.5            # 0..1
    models:
      module: {version: flask-module-v1, sha256: <64 hex>}
      commit: {version: apachejit-jit-v1, sha256: <64 hex>}   # optional: omitted, escalation is not configured
    scoring_timeout_seconds: 60    # > 0
    coverage_refresh_interval: weekly   # must match the coverage workflow's schedule
    stop_at_first_failure: false   # optional, off by default
    run_only_in_budget: false      # optional, off by default

validate() never raises. It returns the parsed settings and a list of problems; any problem means the
CI hook warns and runs the default test order (section 3.3.2). Every plan records the settings used.
"""

from __future__ import annotations

import re

import yaml

UNITS = {"artefacts", "lines"}
KEYS = {"default", "effort"}
INTERVALS = {"daily": "0 3 * * *", "weekly": "0 3 * * 1"}  # the coverage workflow's cron per interval
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load(path: str) -> tuple[dict, list[str]]:
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        return {}, [f"settings file unreadable: {e}"]
    return raw, validate(raw)


def validate(s: dict, registry_versions: set[str] | None = None, coverage_cron: str | None = None) -> list[str]:
    p: list[str] = []
    b = s.get("budget")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0 < b <= 100:
        p.append("budget must be a number above 0 and at most 100 (percent)")
    if s.get("budget_unit") not in UNITS:
        p.append(f"budget_unit must be one of {sorted(UNITS)}")
    if s.get("ranking_key") not in KEYS:
        p.append(f"ranking_key must be one of {sorted(KEYS)}")
    t = s.get("risk_threshold")
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not 0 <= t <= 1:
        p.append("risk_threshold must be between 0 and 1")
    models = s.get("models") or {}
    for g in ("module", "commit"):
        if g == "commit" and not models.get("commit"):
            continue  # no just-in-time model: commits are not scored and build escalation is not configured
        m = models.get(g) or {}
        if not m.get("version"):
            p.append(f"models.{g}.version is missing")
        elif registry_versions is not None and m["version"] not in registry_versions:
            p.append(f"models.{g}.version {m['version']!r} is not published in the registry")
        if not HEX64.match(str(m.get("sha256", ""))):
            p.append(f"models.{g}.sha256 must be a 64-character hex checksum")
    to = s.get("scoring_timeout_seconds")
    if not isinstance(to, (int, float)) or isinstance(to, bool) or to <= 0:
        p.append("scoring_timeout_seconds must be above zero")
    iv = s.get("coverage_refresh_interval")
    if iv not in INTERVALS:
        p.append(f"coverage_refresh_interval must be one of {sorted(INTERVALS)}")
    elif coverage_cron is not None and INTERVALS[iv] != coverage_cron.strip():
        p.append(f"coverage_refresh_interval {iv!r} does not match the coverage workflow schedule {coverage_cron!r}")
    for k in ("stop_at_first_failure", "run_only_in_budget"):
        if k in s and not isinstance(s[k], bool):
            p.append(f"{k} must be true or false")
    return p
