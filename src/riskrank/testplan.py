"""Ordered test plan from a budgeted module ranking (FR17, FR18, NFR17). Module level only.

ranked        modules touched by the change, best first, each with `in_budget`
zero_size     touched modules excluded from ranking because their size is zero (FR9)
coverage      {test id: set of modules it exercises}, from the last published coverage map
current       every test the repository has now (so newly added tests are never lost)

Ordering (FR18):
  * a test's key is the ranks of the ranked modules it exercises, best first; keys compare element by
    element, and a key that runs out first sorts AFTER a tied key that continues;
  * remaining ties: ascending test id;
  * a test is in budget when, and only when, the module that sets its rank is in budget;
  * all other tests form one trailing block in ascending id order, counted in three groups:
    zero_size_only, no_coverage_record and no_ranked_module.
No test is ever removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlanItem:
    test: str
    position: int
    in_budget: bool
    deciding_module: str | None
    group: str  # "ranked" | "zero_size_only" | "no_coverage_record" | "no_ranked_module"


@dataclass
class TestPlan:
    items: list[PlanItem]
    stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"tests": [vars(i) for i in self.items], "stats": self.stats}


def build_plan(
    ranked: list[tuple[str, bool]],
    zero_size: set[str],
    coverage: dict[str, set[str]],
    current: list[str],
    durations: dict[str, float] | None = None,
    map_commit: str | None = None,
    commits_since_map: int | None = None,
    map_age_intervals: float | None = None,
) -> TestPlan:
    rank = {m: i for i, (m, _) in enumerate(ranked)}
    budget = dict(ranked)
    durations = durations or {}

    keyed, trailing = [], []
    for t in current:
        if t not in coverage:
            trailing.append((t, "no_coverage_record"))
            continue
        hits = sorted(rank[m] for m in coverage[t] if m in rank)
        if not hits:
            group = "zero_size_only" if coverage[t] & zero_size else "no_ranked_module"
            trailing.append((t, group))
            continue
        # pad with +inf so that a key which runs out sorts after a tied key that continues
        keyed.append((tuple(hits), t))
    width = max((len(k) for k, _ in keyed), default=0)
    keyed.sort(key=lambda kt: (kt[0] + (math.inf,) * (width - len(kt[0])), kt[1]))
    trailing.sort(key=lambda tg: tg[0])

    items: list[PlanItem] = []
    for key, t in keyed:
        decider = ranked[key[0]][0]
        items.append(PlanItem(t, len(items) + 1, bool(budget[decider]), decider, "ranked"))
    for t, g in trailing:
        items.append(PlanItem(t, len(items) + 1, False, None, g))

    in_budget = [i.test for i in items if i.in_budget]
    timed = [t for t in current if t in durations]
    total = sum(durations[t] for t in timed)
    covered = set().union(*coverage.values()) if coverage else set()
    stats = {
        "tests_total": len(items),
        "in_budget_count": len(in_budget),
        "in_budget_fraction": len(in_budget) / len(items) if items else 0.0,
        "in_budget_runtime_share": (sum(durations.get(t, 0.0) for t in in_budget) / total) if total else None,
        "tests_without_duration": len(current) - len(timed),
        "trailing": {g: sum(i.group == g for i in items)
                     for g in ("zero_size_only", "no_coverage_record", "no_ranked_module")},
        "ranked_modules_missing_from_map": sorted(m for m, _ in ranked if m not in covered),
        "coverage_map_commit": map_commit,
        "commits_since_map": commits_since_map,
        "stale_map_warning": bool(map_age_intervals is not None and map_age_intervals > 2),
    }
    assert len(items) == len(set(current)), "a test was lost or duplicated"
    return TestPlan(items, stats)
