"""The prioritisation dashboard (FR11, FR15, FR16; NFR8, NFR9, NFR15), laid out as in Figure 3.6.

    RISKRANK_DB=sqlite:///riskrank.db RISKRANK_AUTH=~/.config/riskrank/dashboard-auth.json \
        streamlit run src/riskrank/dashboard.py --server.address 127.0.0.1

Nothing is read from the plan store until the login has succeeded (NFR15). The settings bar previews
other settings on a stored run's scores with the prioritisation engine's own code (budget.rank and
budget.mark); the pipeline keeps using the repository's settings file. The test-plan region always shows
the plan the pipeline built, because the store holds no coverage map to rebuild it from.
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from riskrank import auth
from riskrank.budget import Z, mark, rank, realised_share
from riskrank.store import Store

FEATURE_LABELS = {
    "size": "lines of code", "complexity": "cyclomatic complexity", "churn": "code churn in this module",
    "changes": "number of changes to this module", "prior_fixes": "prior fix history of this module",
    "la": "lines added", "ld": "lines deleted", "nf": "files touched", "ns": "subsystems touched",
    "nd": "directories touched", "ent": "change entropy", "ndev": "developers on the touched files",
    "age": "time since the files last changed", "nuc": "unique changes to the files",
    "aexp": "author experience", "arexp": "author recent experience", "asexp": "author subsystem experience",
}
BASELINE_ORDER = ["system", "random", "largest-first", "smallest-first", "recency-based"]


def login() -> str:
    """Return the signed-in user, or show the login form and stop before any content."""
    if st.session_state.get("user"):
        return st.session_state["user"]
    record = auth.load(os.path.expanduser(os.environ.get("RISKRANK_AUTH", "~/.config/riskrank/dashboard-auth.json")))
    st.title("RiskRank dashboard")
    if record is None:
        st.error("No login is configured. Run `python -m riskrank.auth` on the workstation first.")
        st.stop()
    with st.form("login"):
        user = st.text_input("User name")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if auth.verify(record, user, pw):
                st.session_state["user"] = user
                st.rerun()
            st.error("User name or password not recognised.")
    st.stop()


@st.cache_resource
def store(url: str) -> Store:
    return Store(url)


def escalation_text(run: dict, commits: pd.DataFrame) -> str:
    """The pipeline's escalation decision, with the deciding value unrounded beside the threshold it was
    compared with (walkthrough HD-1, HD-2, HD-10). The settings bar does not change it."""
    status = run["jit_status"]
    if status == "not configured":
        return "Build escalation: not configured"
    if status != "assessed":
        return f"Build escalated: not assessed ({status})"
    inside = commits[commits["in_training_range"].astype(bool)] if len(commits) else commits
    thr = float(run["risk_threshold"] or 0.5)
    top = inside["p"].max() if len(inside) else None
    cmp = "" if top is None else f" (highest commit risk {top:.4f} {'>' if top > thr else '≤'} threshold {thr:.2f})"
    return (f"Build escalated: **{'yes' if run['escalated'] else 'no'}**{cmp} · decided by the pipeline with "
            "`riskrank.yml`'s threshold, counting only commits whose features lie inside the model's training range")


def reason_chart(codes: list[dict]) -> alt.Chart:
    d = pd.DataFrame(codes)
    d["feature"] = d["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    d["effect"] = d["direction"].map({"raised": "raises risk", "lowered": "lowers risk"})
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("contribution:Q", title="contribution to the uncalibrated score"),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                    range=["#1f3a5f", "#b5834c"]), legend=None),
        tooltip=["feature", "value", "contribution", "effect"]).properties(height=44 * len(d) + 30)


def explanation_panel(title: str, p: float, rank_no: int | None, codes: list[dict], model: dict | None) -> None:
    st.markdown("#### Why this risk score?")
    st.markdown(f"Selected: `{title}` · risk **{p:.2f}**" + (f" · rank {rank_no}" if rank_no else ""))
    if not codes:
        st.info("No reason codes stored for this score.")
        return
    st.altair_chart(reason_chart(codes), width="stretch")
    st.caption("Bars show feature contributions to the model's **uncalibrated** score: dark raises risk, "
               "light lowers it. They show what the model associates with risk, not what causes defects.")
    attribution = ((model or {}).get("record") or {}).get("explanation", {}).get("attribution", "")
    note = "Contributions assume the features are independent."
    if "kernel" in attribution.lower():
        note += " Explanations for this model are approximate."
    st.caption(f"Attribution: {attribution or 'not recorded'}. {note}")


def main() -> None:
    st.set_page_config(page_title="RiskRank dashboard", layout="wide")
    user = login()
    db = store(os.environ.get("RISKRANK_DB", "sqlite:///riskrank.db"))

    projects = db.projects()
    if not projects:
        st.title("Regression-Test-Prioritisation Dashboard")
        st.info("No pipeline run has been ingested yet. Run `python -m riskrank.ingest --repo <owner/name>`.")
        st.stop()

    # ---------------------------------------------------------------- header
    top_l, top_r = st.columns([3, 2])
    with top_r:
        project = st.selectbox("Project", projects)
        runs = db.runtime_runs(project)
        labels = {f"{str(r.created)[:16].replace('T', ' ')} · {str(r.commit_after or '')[:7]} · run {r.build_ref}": int(r.id)
                  for r in runs.itertuples()}
        run_id = labels[st.selectbox("Build", list(labels))]
    run = db.run(run_id)
    ranking = db.ranking(run_id)
    commits = db.commit_scores(project)
    commits_this = commits[commits["build_ref"] == run["build_ref"]]
    # the module model the build was configured with (a build that touched no module scored nothing)
    module_model = run["module_model"] or (((run.get("detail") or {}).get("settings") or {}).get("models") or {}) \
        .get("module", {}).get("version")
    model = db.model(module_model)
    view_is_commits = st.session_state.get("granularity") == "commits"
    shown_model = (run["jit_model"] or "none") if view_is_commits else (module_model or "none")
    with top_l:
        st.markdown("## Regression-Test-Prioritisation Dashboard")
        st.markdown(f"Project: `{project}` · {'Commit' if view_is_commits else 'Module'} model: `{shown_model}` "
                    f"(calibrated) · Build run {run['build_ref']} · commit `{str(run['commit_after'] or '')[:7]}`")
        st.markdown(f"{escalation_text(run, commits_this)} · signed in: **{user}**")
    corpus_note = {False: run["transferred"], True: (run.get("detail") or {}).get("commit_transferred", True)}
    if corpus_note[view_is_commits]:
        st.warning(f"**Scores transferred:** `{shown_model}` was trained on another corpus "
                   f"({'Java projects, ApacheJIT' if view_is_commits else 'pallets/flask'}), not on `{project}`.")
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    if run["status"] == "default order":
        st.warning(f"This build ran the default test order: {run['failure_reason']}")
    elif run["status"] == "partial":
        st.warning("Partial run: the module plan was built, but commit-level risk was not assessed.")
    if run["stale_map"]:
        st.warning("The coverage map is more than two refresh intervals old. The plan was still built from it.")

    # ---------------------------------------------------------------- settings bar
    stored = {"ranking_key": run["ranking_key"], "budget_unit": run["budget_unit"],
              "budget": float(run["budget"] or 20), "risk_threshold": float(run["risk_threshold"] or 0.5)}
    c = st.columns([1.2, 1.4, 1.2, 1.2, 1.2])
    granularity = c[0].selectbox("Granularity", ["modules", "commits"], key="granularity")
    commit_view = granularity == "commits"
    keys = {"calibrated risk": "default", "effort-normalised": "effort"}
    units = {"modules (artefacts)": "artefacts", "lines of code": "lines"}
    key = keys[c[1].selectbox("Ranking key", list(keys), index=list(keys.values()).index(stored["ranking_key"]),
                              disabled=commit_view)]
    unit = units[c[2].selectbox("Budget unit", list(units), index=list(units.values()).index(stored["budget_unit"]),
                                disabled=commit_view)]
    budget = c[3].number_input("Budget value (%)", 1.0, 100.0, stored["budget"], 1.0, disabled=commit_view)
    threshold = c[4].number_input("Risk threshold", 0.0, 1.0, stored["risk_threshold"], 0.05)
    preview = (key, unit, budget, threshold) != (stored["ranking_key"], stored["budget_unit"], stored["budget"],
                                                 stored["risk_threshold"])
    if commit_view:
        st.caption("Budget controls apply to module plans only; the commit view shows stored historical commit scores.")
    elif preview:
        st.info("**Preview**: the ranking below uses these settings on the stored scores. The pipeline still uses "
                "the repository's settings file; edit `riskrank.yml` to apply them.")

    # ---------------------------------------------------------------- ranked plan + explanation
    left, right = st.columns([3, 2])
    if commit_view:
        with left:
            st.markdown("#### Stored commit risk scores (historical, not a live ranking)")
            if commits.empty:
                st.info("No commit has been scored in this project "
                        + ("(no just-in-time model is configured)." if run["jit_status"] == "not configured" else "."))
                sel = None
            else:
                every = st.checkbox("Show commits from every build", value=False, key="all_builds")
                shown = (commits if every else commits_this).reset_index(drop=True)
                if shown.empty:
                    st.info("This build scored no commit. Tick the box above to see earlier builds.")
                    sel = None
                else:
                    view = shown.assign(commit=shown["commit"].str[:10], risk=shown["p"].map(lambda v: f"{v:.4f}"),
                                        flagged=shown["p"] > threshold,
                                        features_outside_training_range=~shown["in_training_range"].astype(bool))
                    ev = st.dataframe(view[["commit", "risk", "flagged", "features_outside_training_range", "created",
                                            "model_version"]].rename(columns={
                                                "created": "scored in build at", "model_version": "model",
                                                "features_outside_training_range": "features outside training range"}),
                                      on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch",
                                      key="commits")
                    sel = ev.selection.rows[0] if ev.selection.rows else 0
                st.caption("Two different checks: the model was trained on another corpus (Java, ApacheJIT), and, "
                           "separately, a commit's features may fall outside the range seen in training. Escalation "
                           "counts only commits whose features are inside that range.")
        with right:
            if sel is not None:
                r = shown.iloc[sel]
                explanation_panel(r["commit"][:10], r["p"], None, r["reason_codes"] or [], db.model(r["model_version"]))
    else:
        if len(ranking):
            r = rank(ranking.rename(columns={"module": "id"}), key)
            r["in_budget"] = mark(r, unit, budget / 100, key).to_numpy()
            r["flagged"] = r["p"] > threshold
            share = realised_share(r, r["in_budget"].astype(bool), unit)
            r.insert(0, "#", range(1, len(r) + 1))
        else:
            r, share = ranking.rename(columns={"module": "id"}), None
        with left:
            st.markdown("#### Risk-ranked modules changed by this build")
            k = int(r["in_budget"].sum()) if len(r) else 0
            last_in = int(r.loc[r["in_budget"].astype(bool), "#"].max()) if k else 0
            st.markdown(f"{(run.get('detail') or {}).get('modules_touched', len(r))} modules touched · "
                        f"{budget:g}% budget ({'of lines' if unit == 'lines' else 'of modules'}) covers "
                        + (f"ranks 1–{last_in}" if k else "no module")
                        + (f" · realised share {share:.1%}" if share is not None else "")
                        + (" (the module at the boundary is included whole, so the share can exceed the budget)"
                           if share is not None and unit == "lines" and share > budget / 100 + 1e-9 else ""))
            sel = None
            if len(r):
                unmapped = set(((run.get("detail") or {}).get("stats") or {}).get("ranked_modules_missing_from_map", []))
                r["tests mapped"] = ["no tests mapped" if m in unmapped else "yes" for m in r["id"]]
                r["size"] = r["size"].round().astype("Int64")
                show = r[["#", "p", "in_budget", "flagged", "tests mapped", "id", "size"]].rename(
                    columns={"id": "module", "p": "risk", "size": "lines of code", "in_budget": "in budget"})
                if key == "effort":
                    show.insert(4, "lower-bounded risk per 1,000 lines",
                                (r["p"] * (1 - Z) + Z) / r["size"].where(r["size"] > 0) * 1000)
                styled = show.style.apply(lambda row: ["background-color: rgba(31,58,95,0.10)" if row["in budget"]
                                                       else "" for _ in row], axis=1).format({"risk": "{:.2f}"})
                ev = st.dataframe(styled, on_select="rerun", selection_mode="single-row", hide_index=True,
                                  width="stretch", key="modules")
                clicked = ev.selection.rows[0] if ev.selection.rows else 0
                names = [f"{int(n)}. {m}" for n, m in zip(r["#"], r["id"])]
                sel = names.index(st.selectbox("Explain module (keyboard: Tab here, then arrow keys)", names,
                                               index=clicked, key=f"explain-{run_id}-{key}-{unit}"))
                st.caption(f"Shaded rows = inside the budget. Flagged = risk above the {threshold:.2f} threshold"
                           + (f"; lower-bounded risk = {1 - Z:.2f} × risk + {Z:.2f}." if key == "effort" else ".")
                           + " Select a row to see its reasons.")
            else:
                st.info("This build touched no source module, so nothing was ranked. All tests ran in the trailing block.")
        with right:
            if sel is not None:
                row = r.iloc[sel]
                explanation_panel(row["id"], row["p"], int(row["#"]), row["reason_codes"] or [], model)

    # ---------------------------------------------------------------- ordered test plan
    tp = db.test_plan(run_id)
    stats = (run.get("detail") or {}).get("stats") or {}
    st.markdown("#### Ordered test plan: what the pipeline runs")
    st.caption("Built by the pipeline from the stored module ranking and settings"
               + (", not from the preview above." if preview else "."))
    if tp.empty:
        st.info("No test plan stored for this build.")
    else:
        n_in = int(tp["in_budget"].sum())
        stored_rank = dict(zip(ranking["module"], ranking["rank"]))
        tp["from rank"] = tp["deciding_module"].map(stored_rank).astype("Int64")
        unit_word = "of the changed lines" if stored["budget_unit"] == "lines" else "of the changed modules"
        st.markdown(f"The {stored['budget']:g}% budget ({unit_word}) selects ranks "
                    + (f"1–{int(ranking.loc[ranking['in_budget'].astype(bool), 'rank'].max())}"
                       if "in_budget" in ranking and ranking["in_budget"].astype(bool).any() else "none")
                    + f" · tests that exercise them: **{n_in}** of {len(tp)} ({n_in / len(tp):.1%} of the suite)")
        missing_now = (stats.get("ranked_modules_missing_from_map") or [])
        in_budget_mods = set(ranking.loc[ranking["in_budget"].astype(bool), "module"]) if "in_budget" in ranking else set()
        if n_in == 0:
            no_tests = sorted(in_budget_mods & set(missing_now))
            st.warning("**No test is inside the budget.** "
                       + (f"The in-budget module{'s' if len(no_tests) > 1 else ''} {', '.join(f'`{m}`' for m in no_tests)} "
                          "has no test in the coverage map (new or untested code), " if no_tests else
                          "No test in the coverage map exercises the in-budget modules, ")
                       + "so the plan starts with the tests of the next-ranked modules. Every test still runs.")
        age = stats.get("commits_since_map")
        if age is not None and age >= 10:
            st.warning(f"The coverage map is {age} commits old: tests for code added since then are not mapped.")
        styled = (tp[["position", "test", "deciding_module", "from rank", "in_budget"]]
                  .rename(columns={"position": "#", "deciding_module": "module that sets its rank",
                                   "in_budget": "in budget"})
                  .style.apply(lambda row: ["background-color: rgba(31,58,95,0.10)" if row["in budget"] else ""
                                            for _ in row], axis=1))
        st.dataframe(styled, hide_index=True, width="stretch", height=300)
        trailing = stats.get("trailing", {})
        missing = stats.get("ranked_modules_missing_from_map", [])
        st.markdown(
            f"Coverage map at commit `{str(run.get('coverage_map_commit') or 'unknown')[:7]}`"
            + (f" ({stats['commits_since_map']} commits ago)" if stats.get("commits_since_map") is not None else "")
            + f" · **{len(missing)}** ranked module(s) not in the map"
            + (f" ({', '.join(missing)})" if missing else "")
            + f" · unranked tests: {trailing.get('no_ranked_module', 0)} exercise no ranked module, "
            f"{trailing.get('no_coverage_record', 0)} have no coverage record, "
            f"{trailing.get('zero_size_only', 0)} excluded for zero size")

    # ---------------------------------------------------------------- evaluation panels
    off = db.latest_offline(module_model)
    st.markdown("### Offline model evaluation")
    st.info(f"These panels evaluate `{module_model}` offline on **{(off or {}).get('corpus') or 'its evaluation corpus'}**, "
            f"not on `{project}`'s builds, and do not change with the settings above.")
    e1 = st.container()  # the cost-effectiveness curve on its own full-width row (walkthrough finding)
    e2, e3 = st.columns(2)
    with e1:
        st.markdown("#### Cost-effectiveness (offline, lines-of-code budget)")
        if not off:
            st.info(f"No offline evaluation is stored for `{module_model}` yet.")
        else:
            st.caption(f"`{module_model}` · corpus: {off['corpus']} · evaluated {str(off['created'])[:10]} · "
                       "default key, independent of the settings above")
            curves = (off.get("detail") or {}).get("curves", {})
            d = pd.DataFrame([{"ordering": name, "lines inspected (%)": x * 100, "defect recall (%)": y * 100}
                              for name, pts in curves.items() for x, y in pts])
            if len(d):
                line = alt.Chart(d).mark_line().encode(
                    x="lines inspected (%):Q", y="defect recall (%):Q",
                    strokeDash=alt.StrokeDash("ordering:N", sort=BASELINE_ORDER),
                    color=alt.Color("ordering:N", sort=BASELINE_ORDER))
                rule = alt.Chart(pd.DataFrame({"x": [20]})).mark_rule(color="#b5834c").encode(x="x:Q")
                st.altair_chart(line + rule, width="stretch")
            vals = [("system", off["recall"])] + [(b["baseline"], b["recall"]) for b in off["baselines"]
                                                   if b["recall"] is not None]
            best = max(vals, key=lambda v: v[1])
            st.markdown("Defect recall at the 20% budget: " + " · ".join(f"{n} {v:.1%}" for n, v in vals)
                        + f" (highest: {best[0]})")
    with e2:
        st.markdown("#### Calibration: predicted against observed")
        bins = ((off or {}).get("detail") or {}).get("reliability", [])
        if not bins:
            st.info("No reliability data stored for this model version yet.")
        else:
            st.caption(f"Brier score: {off['detail'].get('brier', float('nan')):.3f} · grey diagonal = perfect calibration")
            b = pd.DataFrame(bins)
            pts = alt.Chart(b).mark_line(point=True).encode(
                x=alt.X("mean_p:Q", title="predicted risk", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed:Q", title="observed defect rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=["mean_p", "observed", "n"])
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="grey").encode(x="x", y="y")
            st.altair_chart(diag + pts, width="stretch")
    with e3:
        st.markdown("#### Risk trend over recent builds")
        trend = db.risk_trend(project, limit=12)
        if trend.empty:
            st.info("No build with in-budget modules yet.")
        else:
            st.caption(f"Average risk of in-budget modules, last {len(trend)} builds with a ranking")
            trend["build"] = trend["created"].astype(str).str[:16] + " " + trend["commit_after"].str[:7]
            st.altair_chart(alt.Chart(trend).mark_line(point=True).encode(
                x=alt.X("build:N", sort=None, axis=alt.Axis(labels=False)), y=alt.Y("mean_in_budget_risk:Q", title="average risk"),
                tooltip=["build", "mean_in_budget_risk", "in_budget_modules"]), width="stretch")


main()
