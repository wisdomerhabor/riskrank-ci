"""The plan store behind its storage interface (Figure 3.4; FR10, FR15; NFR11).

    store = Store("sqlite:///riskrank.db")            # workstation default
    store = Store("postgresql+psycopg://u:p@h/db")     # the same application logic on PostgreSQL

Every caller (plan-ingest, offline evaluation writer, dashboard) goes through the methods of `Store`;
none writes SQL. The tables follow the entity-relationship model: corpus, model version, artefact,
prediction, prioritisation run (runtime or offline), ranked item, baseline result, test, coverage map
and test-plan item, plus an ingest log that records stored, rejected, missing and lost runs.
Only the main fields are columns; the rest of each record is kept as JSON beside them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, UniqueConstraint, create_engine, event, func, select)

meta = MetaData()

corpus = Table("corpus", meta,
               Column("name", String, primary_key=True),
               Column("projects", JSON))

model_version = Table("model_version", meta,
                      Column("version", String, primary_key=True),
                      Column("granularity", String, nullable=False),
                      Column("algorithm", String),
                      Column("corpus", JSON),
                      Column("training_window", JSON),
                      Column("bundle_sha256", String),
                      Column("published", String),
                      Column("record", JSON))  # the full model metadata from the release asset

artefact = Table("artefact", meta,
                 Column("id", Integer, primary_key=True),
                 Column("project", String, nullable=False),
                 Column("kind", String, nullable=False),        # module | commit
                 Column("name", String, nullable=False),
                 Column("at_commit", String, nullable=False),   # snapshot or commit the record describes
                 Column("size", Float),
                 Column("size_unit", String),                   # lines | churn
                 UniqueConstraint("project", "kind", "name", "at_commit"))

prioritisation_run = Table("prioritisation_run", meta,
                           Column("id", Integer, primary_key=True),
                           Column("run_type", String, nullable=False),     # runtime | offline
                           Column("project", String, nullable=False),
                           Column("build_ref", String),                    # CI run id (runtime)
                           Column("commit_after", String),
                           Column("commit_before", String),
                           Column("created", String),
                           Column("status", String),
                           Column("failure_reason", String),
                           Column("module_model", String, ForeignKey("model_version.version")),
                           Column("jit_model", String),
                           Column("transferred", Boolean),
                           Column("ranking_key", String),
                           Column("budget_unit", String),
                           Column("budget", Float),
                           Column("risk_threshold", Float),
                           Column("tie_break", String),
                           Column("escalated", Boolean),
                           Column("jit_status", String),
                           Column("realised_share", Float),
                           Column("coverage_map_id", Integer, ForeignKey("coverage_map.id")),
                           Column("stale_map", Boolean),
                           Column("zero_size_excluded", Integer),
                           Column("corpus", String),                       # offline: evaluation corpus
                           Column("recall", Float),                        # offline: system recall at budget
                           Column("detail", JSON),                         # timings, counts, curves, bins
                           UniqueConstraint("run_type", "project", "build_ref"))

prediction = Table("prediction", meta,
                   Column("id", Integer, primary_key=True),
                   Column("artefact_id", Integer, ForeignKey("artefact.id"), nullable=False),
                   Column("model_version", String, nullable=False),
                   Column("run_id", Integer, ForeignKey("prioritisation_run.id")),
                   Column("p", Float, nullable=False),
                   Column("raw", Float),
                   Column("reason_codes", JSON),
                   Column("in_training_range", Boolean))

ranked_item = Table("ranked_item", meta,
                    Column("run_id", Integer, ForeignKey("prioritisation_run.id"), primary_key=True),
                    Column("rank", Integer, primary_key=True),
                    Column("prediction_id", Integer, ForeignKey("prediction.id"), nullable=False),
                    Column("in_budget", Boolean),
                    Column("flagged", Boolean))

baseline_result = Table("baseline_result", meta,
                        Column("run_id", Integer, ForeignKey("prioritisation_run.id"), primary_key=True),
                        Column("baseline", String, primary_key=True),
                        Column("recall", Float),
                        Column("permutations", Integer),
                        Column("interval_low", Float),
                        Column("interval_high", Float))

test = Table("test", meta,
             Column("id", Integer, primary_key=True),
             Column("project", String, nullable=False),
             Column("name", String, nullable=False),
             UniqueConstraint("project", "name"))

coverage_map = Table("coverage_map", meta,
                     Column("id", Integer, primary_key=True),
                     Column("project", String, nullable=False),
                     Column("commit", String, nullable=False),
                     UniqueConstraint("project", "commit"))

test_plan_item = Table("test_plan_item", meta,
                       Column("run_id", Integer, ForeignKey("prioritisation_run.id"), primary_key=True),
                       Column("position", Integer, primary_key=True),
                       Column("test_id", Integer, ForeignKey("test.id"), nullable=False),
                       Column("in_budget", Boolean),
                       Column("deciding_module", String),
                       Column("group", String),
                       Column("duration", Float))

ingest_log = Table("ingest_log", meta,
                   Column("id", Integer, primary_key=True),
                   Column("project", String, nullable=False),
                   Column("build_ref", String, nullable=False),
                   Column("outcome", String, nullable=False),  # stored | rejected | missing | lost | retry
                   Column("reason", String),
                   Column("at", DateTime(timezone=True), nullable=False))

TIE_BREAK = "uncalibrated score, highest first; then ascending identifier"


class Store:
    def __init__(self, url: str = "sqlite:///riskrank.db"):
        self.engine = create_engine(url)
        if self.engine.dialect.name == "sqlite":  # enforce foreign keys as PostgreSQL does (NFR11)
            event.listen(self.engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
        meta.create_all(self.engine)

    # ---------------------------------------------------------------- writes
    def put_model(self, record: dict) -> None:
        """Copy a published model version's metadata (from its release asset) into the store."""
        row = {"version": record["version"], "granularity": record["granularity"],
               "algorithm": record.get("algorithm"), "corpus": record.get("corpus"),
               "training_window": record.get("training_window"), "bundle_sha256": record.get("bundle_sha256"),
               "published": record.get("published"), "record": record}
        with self.engine.begin() as c:
            if c.execute(select(model_version.c.version).where(model_version.c.version == row["version"])).first():
                c.execute(model_version.update().where(model_version.c.version == row["version"]).values(**row))
            else:
                c.execute(model_version.insert().values(**row))

    def has_run(self, project: str, build_ref: str) -> bool:
        with self.engine.connect() as c:
            return c.execute(select(prioritisation_run.c.id).where(
                prioritisation_run.c.run_type == "runtime", prioritisation_run.c.project == project,
                prioritisation_run.c.build_ref == build_ref)).first() is not None

    def log_ingest(self, project: str, build_ref: str, outcome: str, reason: str | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(ingest_log.insert().values(project=project, build_ref=build_ref, outcome=outcome,
                                                 reason=reason, at=datetime.now(timezone.utc)))

    def ingest_outcomes(self, project: str) -> pd.DataFrame:
        with self.engine.connect() as c:
            return pd.read_sql(select(ingest_log).where(ingest_log.c.project == project)
                               .order_by(ingest_log.c.id), c)

    def put_runtime_run(self, project: str, build_ref: str, created: str, plan: dict,
                        durations: dict[str, float] | None = None) -> int:
        """Store one runtime prioritisation run from a validated plan file (FR10). Returns its id."""
        durations = durations or {}
        s = plan.get("settings", {})
        complete = plan["status"] in ("complete", "partial")
        stats = plan.get("test_plan", {}).get("stats", {}) if complete else {}
        jit = plan.get("commit_level", {}) if complete else {}
        after = plan.get("push", {}).get("after")
        with self.engine.begin() as c:
            # a run may name a model whose release record could not be read yet: keep a placeholder row,
            # filled in later by put_model
            for version, gran in [(plan.get("module_model"), "module"), (jit.get("model_version"), "commit")]:
                if version and not c.execute(select(model_version.c.version)
                                             .where(model_version.c.version == version)).first():
                    c.execute(model_version.insert().values(version=version, granularity=gran))
            map_id = None
            if stats.get("coverage_map_commit"):
                map_id = self._get_or_add(c, coverage_map, project=project, commit=stats["coverage_map_commit"])
            run_id = c.execute(prioritisation_run.insert().values(
                run_type="runtime", project=project, build_ref=build_ref, created=created,
                commit_after=after, commit_before=plan.get("push", {}).get("before"),
                status=plan["status"],
                failure_reason=plan.get("reason"),
                module_model=plan.get("module_model"), jit_model=jit.get("model_version"),
                transferred=plan.get("module_transferred"),
                ranking_key=s.get("ranking_key"), budget_unit=s.get("budget_unit"), budget=s.get("budget"),
                risk_threshold=s.get("risk_threshold"), tie_break=TIE_BREAK,
                escalated=plan.get("escalated") if jit.get("status") != "not configured" else None,
                jit_status=jit.get("status"),
                realised_share=plan.get("realised_share"), coverage_map_id=map_id,
                stale_map=stats.get("stale_map_warning"),
                zero_size_excluded=sum(1 for r in plan.get("ranking", []) if r["size"] <= 0),
                detail={"stats": stats, "timing_seconds": plan.get("timing_seconds"),
                        "modules_touched": plan.get("modules_touched"),
                        "modules_unparsable": plan.get("modules_unparsable"),
                        "push_commits": plan.get("push", {}).get("commits"),
                        "commit_transferred": jit.get("transferred"),
                        "settings": s}).returning(prioritisation_run.c.id)).scalar_one()
            for r in plan.get("ranking", []):
                a = self._get_or_add(c, artefact, project=project, kind="module", name=r["module"],
                                     at_commit=after, extra={"size": r["size"], "size_unit": "lines"})
                pid = c.execute(prediction.insert().values(
                    artefact_id=a, model_version=plan["module_model"], run_id=run_id, p=r["p"], raw=r["raw"],
                    reason_codes=r["reason_codes"]).returning(prediction.c.id)).scalar_one()
                c.execute(ranked_item.insert().values(run_id=run_id, rank=r["rank"], prediction_id=pid,
                                                      in_budget=r["in_budget"], flagged=r["flagged"]))
            for cm in jit.get("commits", []):
                a = self._get_or_add(c, artefact, project=project, kind="commit", name=cm["id"],
                                     at_commit=cm["id"], extra={"size_unit": "churn"})
                c.execute(prediction.insert().values(
                    artefact_id=a, model_version=jit["model_version"], run_id=run_id, p=cm["p"],
                    reason_codes=cm["reason_codes"], in_training_range=cm["in_training_range"]))
            tests = plan["test_plan"]["tests"] if complete else plan.get("tests", [])
            for t in tests:
                tid = self._get_or_add(c, test, project=project, name=t["test"])
                c.execute(test_plan_item.insert().values(
                    run_id=run_id, position=t["position"], test_id=tid, in_budget=t.get("in_budget", False),
                    deciding_module=t.get("deciding_module"), group=t.get("group", "default order"),
                    duration=durations.get(t["test"])))
        return run_id

    def put_offline_run(self, *, project: str, corpus_name: str, corpus_projects: list[str], model: str,
                        created: str, budget_unit: str, budget: float, ranking_key: str, recall: float,
                        baselines: list[dict], detail: dict, build_ref: str) -> int:
        """Store one offline evaluation run with its baseline results (FR13, FR15).

        baselines: [{"baseline", "recall", "permutations", "interval_low", "interval_high"}]; the
        permutation fields stay empty when the result is exact (random baseline, artefacts budget).
        detail carries the cost-effectiveness curves and the reliability bins the dashboard plots.
        """
        with self.engine.begin() as c:
            if not c.execute(select(corpus.c.name).where(corpus.c.name == corpus_name)).first():
                c.execute(corpus.insert().values(name=corpus_name, projects=corpus_projects))
            run_id = c.execute(prioritisation_run.insert().values(
                run_type="offline", project=project, build_ref=build_ref, created=created, status="complete",
                module_model=model if self._granularity(c, model) == "module" else None,
                jit_model=model if self._granularity(c, model) == "commit" else None,
                ranking_key=ranking_key, budget_unit=budget_unit, budget=budget, tie_break=TIE_BREAK,
                corpus=corpus_name, recall=recall, detail=detail).returning(prioritisation_run.c.id)).scalar_one()
            for b in baselines:
                c.execute(baseline_result.insert().values(run_id=run_id, **b))
        return run_id

    # ---------------------------------------------------------------- reads
    def runtime_runs(self, project: str | None = None) -> pd.DataFrame:
        q = select(prioritisation_run, coverage_map.c.commit.label("coverage_map_commit")).select_from(
            prioritisation_run.outerjoin(coverage_map)).where(prioritisation_run.c.run_type == "runtime")
        if project:
            q = q.where(prioritisation_run.c.project == project)
        with self.engine.connect() as c:
            return pd.read_sql(q.order_by(prioritisation_run.c.created.desc(), prioritisation_run.c.id.desc()), c)

    def projects(self) -> list[str]:
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(select(prioritisation_run.c.project).distinct()
                                            .where(prioritisation_run.c.run_type == "runtime")
                                            .order_by(prioritisation_run.c.project))]

    def run(self, run_id: int) -> dict:
        with self.engine.connect() as c:
            row = c.execute(select(prioritisation_run, coverage_map.c.commit.label("coverage_map_commit"))
                            .select_from(prioritisation_run.outerjoin(coverage_map))
                            .where(prioritisation_run.c.id == run_id)).mappings().one()
            return dict(row)

    def ranking(self, run_id: int) -> pd.DataFrame:
        """The stored module ranking of a run: rank, module, p, raw, size, in_budget, flagged, reason_codes."""
        q = (select(ranked_item.c.rank, artefact.c.name.label("module"), prediction.c.p, prediction.c.raw,
                    artefact.c.size, ranked_item.c.in_budget, ranked_item.c.flagged, prediction.c.reason_codes)
             .select_from(ranked_item.join(prediction).join(artefact))
             .where(ranked_item.c.run_id == run_id).order_by(ranked_item.c.rank))
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all(),
                                columns=["rank", "module", "p", "raw", "size", "in_budget", "flagged", "reason_codes"])

    def commit_scores(self, project: str) -> pd.DataFrame:
        """Stored historical commit scores (the commit view shows these, never a live ranking)."""
        q = (select(artefact.c.name.label("commit"), prediction.c.p, prediction.c.raw, prediction.c.model_version,
                    prediction.c.in_training_range, prediction.c.reason_codes, prioritisation_run.c.created,
                    prioritisation_run.c.build_ref)
             .select_from(prediction.join(artefact).join(prioritisation_run))
             .where(artefact.c.kind == "commit", artefact.c.project == project)
             .order_by(prioritisation_run.c.created.desc()))
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all(),
                                columns=["commit", "p", "raw", "model_version", "in_training_range", "reason_codes",
                                         "created", "build_ref"])

    def test_plan(self, run_id: int) -> pd.DataFrame:
        q = (select(test_plan_item.c.position, test.c.name.label("test"), test_plan_item.c.in_budget,
                    test_plan_item.c.deciding_module, test_plan_item.c.group, test_plan_item.c.duration)
             .select_from(test_plan_item.join(test)).where(test_plan_item.c.run_id == run_id)
             .order_by(test_plan_item.c.position))
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all(),
                                columns=["position", "test", "in_budget", "deciding_module", "group", "duration"])

    def risk_trend(self, project: str, limit: int = 30) -> pd.DataFrame:
        """Mean calibrated risk of the in-budget modules, per recent runtime run."""
        q = (select(prioritisation_run.c.id, prioritisation_run.c.created, prioritisation_run.c.commit_after,
                    func.avg(prediction.c.p).label("mean_in_budget_risk"),
                    func.count(prediction.c.id).label("in_budget_modules"))
             .select_from(ranked_item.join(prediction).join(prioritisation_run,
                                                            ranked_item.c.run_id == prioritisation_run.c.id))
             .where(prioritisation_run.c.project == project, prioritisation_run.c.run_type == "runtime",
                    ranked_item.c.in_budget.is_(True))
             .group_by(prioritisation_run.c.id, prioritisation_run.c.created, prioritisation_run.c.commit_after)
             .order_by(prioritisation_run.c.created.desc()).limit(limit))
        with self.engine.connect() as c:
            return pd.read_sql(q, c).sort_values("created")

    def model(self, version: str | None) -> dict | None:
        if not version:
            return None
        with self.engine.connect() as c:
            r = c.execute(select(model_version).where(model_version.c.version == version)).mappings().first()
            return dict(r) if r else None

    def latest_offline(self, model: str | None) -> dict | None:
        """The latest offline evaluation of a model version, with its baselines (FR15)."""
        if not model:
            return None
        with self.engine.connect() as c:
            r = c.execute(select(prioritisation_run).where(
                prioritisation_run.c.run_type == "offline",
                (prioritisation_run.c.module_model == model) | (prioritisation_run.c.jit_model == model))
                .order_by(prioritisation_run.c.created.desc(), prioritisation_run.c.id.desc())).mappings().first()
            if not r:
                return None
            out = dict(r)
            out["baselines"] = [dict(b) for b in c.execute(
                select(baseline_result).where(baseline_result.c.run_id == r["id"])).mappings()]
            return out

    def dump(self) -> dict[str, list[dict]]:
        """Every row of every table, for migration to another database (NFR11)."""
        with self.engine.connect() as c:
            return {t.name: [dict(r) for r in c.execute(select(t)).mappings()] for t in meta.sorted_tables}

    def load(self, rows: dict[str, list[dict]]) -> None:
        with self.engine.begin() as c:
            for t in meta.sorted_tables:
                if rows.get(t.name):
                    c.execute(t.insert(), rows[t.name])
            if self.engine.dialect.name == "postgresql":
                # rows arrive with their ids, so move each id sequence past the largest one copied
                for t in meta.sorted_tables:
                    if "id" in t.c and isinstance(t.c.id.type, Integer) and t.c.id.primary_key:
                        c.exec_driver_sql(f"SELECT setval(pg_get_serial_sequence('{t.name}', 'id'), "
                                          f"COALESCE((SELECT MAX(id) FROM {t.name}), 0) + 1, false)")


    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _get_or_add(c, table: Table, extra: dict | None = None, **key) -> int:
        q = select(table.c.id)
        for k, v in key.items():
            q = q.where(table.c[k] == v)
        hit = c.execute(q).first()
        if hit:
            if extra:
                c.execute(table.update().where(table.c.id == hit[0]).values(**extra))
            return hit[0]
        return c.execute(table.insert().values(**key, **(extra or {})).returning(table.c.id)).scalar_one()

    @staticmethod
    def _granularity(c, version: str) -> str | None:
        r = c.execute(select(model_version.c.granularity).where(model_version.c.version == version)).first()
        return r[0] if r else None


def migrate(source: str, target: str) -> dict[str, int]:
    """Copy every row from one store to another, empty one (NFR11), and check the counts match."""
    rows = Store(source).dump()
    dest = Store(target)
    if any(dest.dump().values()):
        raise SystemExit("target store is not empty")
    dest.load(rows)
    counts = {k: len(v) for k, v in dest.dump().items()}
    assert counts == {k: len(v) for k, v in rows.items()}, "row counts differ after migration"
    return counts


def json_default(o):
    return o.isoformat() if hasattr(o, "isoformat") else str(o)


if __name__ == "__main__":
    # python -m riskrank.store <url>                  -> table row counts
    # python -m riskrank.store migrate <from> <to>    -> copy every row into an empty store (NFR11)
    import sys
    if sys.argv[1:2] == ["migrate"]:
        print(json.dumps(migrate(sys.argv[2], sys.argv[3]), indent=1))
    else:
        st = Store(sys.argv[1] if len(sys.argv) > 1 else "sqlite:///riskrank.db")
        print(json.dumps({k: len(v) for k, v in st.dump().items()}, indent=1))
