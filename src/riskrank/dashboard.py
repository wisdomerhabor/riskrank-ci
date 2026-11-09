"""The prioritisation dashboard (FR11, FR15, FR16; NFR8, NFR9, NFR15), laid out as in Figure 3.6.

    RISKRANK_DB=sqlite:///riskrank.db RISKRANK_AUTH=~/.config/riskrank/dashboard-auth.json \
        streamlit run src/riskrank/dashboard.py --server.address 127.0.0.1

Nothing is read from the plan store until the login has succeeded (NFR15). The settings bar previews
other settings on a stored run's scores with the prioritisation engine's own code (budget.rank and
budget.mark); the pipeline keeps using the repository's settings file. The test-plan region always shows
the plan the pipeline built, because the store holds no coverage map to rebuild it from.
"""

from __future__ import annotations

import html
import os
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from riskrank import auth
from riskrank.budget import Z, mark, rank, realised_share
from riskrank.store import Store

# short axis labels, full names in the tooltip (walkthrough E-2)
FEATURE_LABELS = {
    "size": ("lines of code", "lines of code"), "complexity": ("complexity", "cyclomatic complexity"),
    "churn": ("churn", "code churn in this module"), "changes": ("changes", "number of changes to this module"),
    "prior_fixes": ("prior fixes", "prior fix history of this module"),
    "la": ("lines added", "lines added"), "ld": ("lines deleted", "lines deleted"), "nf": ("files", "files touched"),
    "ns": ("subsystems", "subsystems touched"), "nd": ("directories", "directories touched"),
    "ent": ("entropy", "change entropy"), "ndev": ("developers", "developers on the touched files"),
    "age": ("file age", "time since the files last changed"), "nuc": ("unique changes", "unique changes to the files"),
    "aexp": ("author exp.", "author experience"), "arexp": ("recent exp.", "author recent experience"),
    "asexp": ("subsystem exp.", "author subsystem experience"),
}
ATTRIBUTION_NOTES = {  # one plain-language line per method (E-9)
    "linear": "for a linear model each bar is the feature's weight times its distance from the average module",
    "tree shap": "Tree SHAP shares the score exactly among the features, measured against average background data",
    "kernel": "Kernel SHAP estimates each feature's share by sampling, so the bars are approximate",
}
BASELINE_ORDER = ["system", "random", "largest-first", "smallest-first", "recency-based"]
WAT = timezone(timedelta(hours=1), "WAT")  # Lagos time; Nigeria keeps no daylight saving


def local_time(iso: str | None) -> str | None:
    """'2026-11-08T21:29:24+01:00' -> '8 Nov 2026, 21:29 WAT'; None when missing or unreadable."""
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(WAT)
    except (TypeError, ValueError):
        return None
    return f"{t.day} {t:%b %Y, %H:%M} WAT"


def map_age(stats: dict) -> str:
    """'28 commits ago, committed 8 Nov 2026, 21:29 WAT'; parts that were not recorded are left out."""
    n, when = stats.get("commits_since_map"), local_time(stats.get("coverage_map_committed_at"))
    parts = ([f"{n} commit{'' if n == 1 else 's'} ago"] if n is not None else []) + \
            ([f"committed {when}"] if when else [])
    return ", ".join(parts) or "age not recorded"


PRODUCT = "RiskRank: Regression-Test-Prioritisation Dashboard"  # one name on the tab, the login page and the heading (walkthrough HD-6)


def login() -> str:
    """Return the signed-in user, or show the login form and stop before any content."""
    if st.session_state.get("user"):
        return st.session_state["user"]
    record = auth.load(os.path.expanduser(os.environ.get("RISKRANK_AUTH", "~/.config/riskrank/dashboard-auth.json")))
    st.title(PRODUCT)
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
    cmp = "" if top is None else f" (highest commit-level risk {top:.4f} {'>' if top > thr else '≤'} threshold {thr:.2f})"
    return (f"Build escalated: **{'yes' if run['escalated'] else 'no'}**{cmp} · decided by the pipeline with "
            "`riskrank.yml`'s threshold, counting only commits whose features lie inside the model's training range")


def build_label(r) -> str:
    """A build option with its status, so an escalated or degraded build is found without opening each
    one (walkthrough HD-5)."""
    marks = [m for m, on in (("escalated", r.escalated), ("default order", r.status == "default order"),
                             ("partial", r.status == "partial"), ("stale map", r.stale_map)) if on]
    return (f"{str(r.created)[:16].replace('T', ' ')} · {str(r.commit_after or '')[:7]} · run {r.build_ref}"
            + (f" · {', '.join(marks)}" if marks else ""))


def reading(feature: str, value: float, direction: str) -> str:
    """The tooltip sentence: a zero count reads as 'no prior fixes', not as 'fixes lower risk' (E-5)."""
    short = FEATURE_LABELS.get(feature, (feature, feature))[0]
    effect = {"raised": "raises risk", "lowered": "lowers risk"}.get(direction, "no effect")
    if value == 0 and feature in ("prior_fixes", "changes", "churn", "ndev", "nuc"):
        return f"no {short} (0) {effect}"
    return f"{short} = {value:g} {effect}"


def contribution_domain(code_lists) -> list[float]:
    """One x-axis for every score in view, so bars compare across rows (E-3)."""
    vals = [c["contribution"] for codes in code_lists for c in (codes or [])] or [0.0]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))   # zero always in view; no empty half when all one sign
    pad = 0.05 * ((hi - lo) or 1.0)
    return [lo - (pad if lo < 0 else 0.0), hi + (pad if hi > 0 else 0.0)]


def reason_chart(codes: list[dict], domain: list[float] | None = None) -> alt.Chart:
    d = pd.DataFrame(codes)
    d["reading"] = [reading(f, v, r) for f, v, r in zip(d["feature"], d["value"], d["direction"])]
    d["full name"] = d["feature"].map(lambda f: FEATURE_LABELS.get(f, (f, f))[1])
    d["feature"] = d["feature"].map(lambda f: FEATURE_LABELS.get(f, (f, f))[0])
    d["effect"] = d["direction"].map({"raised": "raises risk", "lowered": "lowers risk"})
    x = alt.X("contribution:Q", title="contribution to the uncalibrated score",
              scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    return alt.Chart(d).mark_bar().encode(
        x=x, y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                    range=["#1f3a5f", "#b5834c"]), legend=None),
        tooltip=["reading", "full name", "contribution"]).properties(height=44 * len(d) + 30)


def explanation_panel(title: str, p: float, rank_no: int | None, codes: list[dict], model: dict | None,
                      raw: float | None = None, domain: list[float] | None = None, default: bool = False,
                      digits: int = 2, top_only: bool = False) -> None:
    st.markdown("## Why this risk score?")
    # a long path wraps instead of breaking mid-word or overflowing; copying it gives the plain path (E-8)
    shown = f'<code style="white-space: normal; overflow-wrap: anywhere">{html.escape(title)}</code>'
    st.markdown(f"Selected: {shown} · risk **{p:.{digits}f}**" + (f" · rank {rank_no}" if rank_no else "")
                + (" (shown by default; select a row to change)" if default else ""),                  # E-6
                unsafe_allow_html=True)
    if not codes:
        st.info("No reason codes stored for this score.")
        return
    st.altair_chart(reason_chart(codes, domain), width="stretch", theme=None)
    if top_only:                                                                                        # C-9
        st.caption(f"Showing the top {len(codes)} features by size of contribution (the pipeline keeps the five "
                   "largest per commit), so the features listed differ between commits.")
    st.caption("Bars show feature contributions to the model's **uncalibrated** score: dark blue raises risk, "
               "tan lowers it. They show what the model associates with risk, not what causes defects.")  # E-7
    record = (model or {}).get("record") or {}
    if raw is not None and not pd.isna(raw) and not top_only:                                         # E-4
        total = sum(c["contribution"] for c in codes)
        scale = (record.get("explanation") or {}).get("scale", "raw")
        st.caption(f"Baseline {raw - total:+.3f} + contributions {total:+.3f} = uncalibrated score {raw:.3f}"
                   f" ({scale}); calibration maps it to the risk {p:.2f} shown above.")
    attribution = (record.get("explanation") or {}).get("attribution", "")
    note = next((v for k, v in ATTRIBUTION_NOTES.items() if k in attribution.lower()), "")
    st.caption(f"Attribution: {attribution or 'not recorded'}" + (f": {note}." if note else ".")
               + " Contributions assume the features are independent.")


def main() -> None:
    st.set_page_config(page_title=PRODUCT, layout="wide", initial_sidebar_state="collapsed")
    # keep the heading levels (HD-8) but at dashboard sizes, and tiles that fit their values (L-4)
    st.markdown("<style>h1 {font-size: 1.75rem !important; padding-top: 0.2rem !important} "
                "h2 {font-size: 1.3rem !important} h3 {font-size: 1.08rem !important} "
                "[data-testid='stMetricValue'], [data-testid='stMetricValue'] * {font-size: 1.15rem !important; "
                "white-space: normal !important; overflow: visible !important; text-overflow: clip !important}"
                "</style>", unsafe_allow_html=True)
    user = login()
    db = store(os.environ.get("RISKRANK_DB", "sqlite:///riskrank.db"))

    projects = db.projects()
    if not projects:
        st.title(PRODUCT)
        st.info("No pipeline run has been ingested yet. Run `python -m riskrank.ingest --repo <owner/name>`.")
        st.stop()

    # ---------------------------------------------------------------- header
    top_l, top_r = st.columns([3, 2])
    with top_r:
        qp = st.query_params
        project = st.selectbox("Project", projects,
                               index=projects.index(qp["project"]) if qp.get("project") in projects else 0)
        runs = db.runtime_runs(project)
        labels = {build_label(r): int(r.id) for r in runs.itertuples()}
        ids = list(labels.values())
        wanted = int(qp["build"]) if str(qp.get("build", "")).isdigit() else None
        run_id = labels[st.selectbox("Build", list(labels), index=ids.index(wanted) if wanted in ids else 0)]
        qp["project"], qp["build"] = project, str(run_id)  # the address bar keeps the choice (HD-9)
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
        st.title(PRODUCT)
        st.markdown(f"{'Commit' if view_is_commits else 'Module'}-level model: `{shown_model}` (calibrated) · "
                    f"commit `{str(run['commit_after'] or '')[:7]}`")
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

    # status tiles that stay above the tabs (walkthrough L-3, L-4)
    t = st.columns(4)
    t[0].metric("Build escalated", {True: "yes", False: "no"}.get(run["escalated"], "not assessed")
                if run["jit_status"] == "assessed" else str(run["jit_status"] or "not assessed"))
    t[1].metric("Pipeline status", str(run["status"] or "unknown"))
    t[2].metric("Coverage map age", map_age(((run.get("detail") or {}).get("stats") or {})))
    t[3].metric("Pushed commit", str(run["commit_after"] or "")[:7] or "–")
    # three labelled sections, each region in a bordered card (L-1, L-2); the settings bar sits in the
    # section it controls (L-5)
    tab_build, tab_plan, tab_off = st.tabs(["This build: ranked changes", "Test plan the pipeline ran",
                                            "Offline model evaluation"])
    with tab_build, st.container(border=True):
            # ---------------------------------------------------------------- settings bar
            stored = {"ranking_key": run["ranking_key"], "budget_unit": run["budget_unit"],
                      "budget": float(run["budget"] or 20), "risk_threshold": float(run["risk_threshold"] or 0.5)}
            # two rows, so no value is cut off on a narrow window (walkthrough S-6)
            row1, row2 = st.columns(3), st.columns(3)
            granularity = row1[0].selectbox("Granularity", ["modules", "commits"], key="granularity",
                                            help="Modules: this build's ranked modules and test plan. Commits: stored "
                                                 "historical commit scores.")
            commit_view = granularity == "commits"
            keys = {"calibrated risk": "default", "effort-normalised": "effort"}
            units = {"modules (artefacts)": "artefacts", "lines of code": "lines"}
            key = keys[row1[1].selectbox(
                "Ranking key", list(keys), index=list(keys.values()).index(stored["ranking_key"]), disabled=commit_view,
                help=f"Calibrated risk: highest probability of a defect first. Effort-normalised: lower-bounded risk "
                     f"({1 - Z:.2f} × risk + {Z:.2f}) per line of code, so small risky modules come first. Affects the "
                     "ranked table only.")]                                                                      # S-2, S-4
            unit = units[row1[2].selectbox(
                "Budget unit", list(units), index=list(units.values()).index(stored["budget_unit"]), disabled=commit_view,
                help="Modules (artefacts): the budget is a share of the changed modules. Lines of code: a share of their "
                     "lines. Affects which rows are in budget.")]                                               # S-2, S-4
            of_what = "of changed lines" if unit == "lines" else "of changed modules"
            budget = row2[0].number_input(f"Budget (% {of_what})", 1.0, 100.0, stored["budget"], 1.0, format="%.0f",
                                          disabled=commit_view, help="Any whole value from 1 to 100. Affects which rows "
                                                                     "are in budget.")                         # S-2, S-7
            threshold = row2[1].number_input("Risk threshold", 0.0, 1.0, stored["risk_threshold"], 0.05, format="%.2f",
                                             help="Changes the flagged column in the ranked table and the commit view. The "
                                                  "header's escalation is the pipeline's decision and does not change.")  # S-2
            row2[0].caption("Valid range 1–100. An invalid entry is not applied.")                                # S-3
            preview = (key, unit, budget, threshold) != (stored["ranking_key"], stored["budget_unit"], stored["budget"],
                                                         stored["risk_threshold"])
            if commit_view:
                st.caption("Budget controls apply to module plans only; the commit view shows stored historical commit scores.")
                if threshold != stored["risk_threshold"]:                                                       # C-4
                    st.info(f"**Preview**: the flagged column uses threshold {threshold:.2f}. The pipeline's escalation "
                            f"used {stored['risk_threshold']:.2f} from `riskrank.yml`.")
            elif preview:
                st.info("**Preview**: the ranking below uses these settings on the stored scores. The pipeline still uses "
                        "the repository's settings file; edit `riskrank.yml` to apply them.")
            else:
                st.caption("These settings match `riskrank.yml` as recorded for this build.")                     # S-1

            # ---------------------------------------------------------------- ranked plan + explanation
            left, right = st.columns([3, 2])
            if commit_view:
                with left:
                    st.markdown("## Stored commit risk scores (historical, not a live ranking)")
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
                            view["created"] = pd.to_datetime(view["created"], utc=True).dt.strftime("%Y-%m-%d %H:%M")  # C-6
                            ev = st.dataframe(view[["commit", "risk", "flagged", "features_outside_training_range", "created",
                                                    "model_version"]].rename(columns={
                                                        "created": "scored in build at", "model_version": "model",
                                                        "features_outside_training_range": "features outside training range"}),
                                              on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch",
                                              key="commits", column_config={"model": st.column_config.TextColumn(width="medium")})
                            sel = ev.selection.rows[0] if ev.selection.rows else 0
                        st.caption("Two different checks: the model was trained on another corpus (Java, ApacheJIT), and, "
                                   "separately, a commit's features may fall outside the range seen in training. Escalation "
                                   "counts only commits whose features are inside that range.")
                with right:
                    if sel is not None:
                        r = shown.iloc[sel]
                        explanation_panel(r["commit"][:10], r["p"], None, r["reason_codes"] or [], db.model(r["model_version"]),
                                          raw=r.get("raw"), domain=contribution_domain(shown["reason_codes"]),
                                          default=not ev.selection.rows, digits=4, top_only=True)
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
                    st.markdown("## Risk-ranked modules changed by this build")
                    k = int(r["in_budget"].sum()) if len(r) else 0
                    last_in = int(r.loc[r["in_budget"].astype(bool), "#"].max()) if k else 0
                    touched = (run.get('detail') or {}).get('modules_touched', len(r))
                    st.markdown(f"{touched} module{'' if touched == 1 else 's'} touched · "
                                f"{budget:g}% budget ({'of lines' if unit == 'lines' else 'of modules'}) covers "
                                + ((f"rank 1" if last_in == 1 else f"ranks 1–{last_in}") if k else "no module")
                                + (f" · realised share {share:.1%}" if share is not None else "")
                                + (" (the module at the boundary is included whole, so the share can exceed the budget)"
                                   if share is not None and unit == "lines" and share > budget / 100 + 1e-9 else ""))
                    if len(r):
                        n_flag = int(r["flagged"].sum())
                        st.caption(f"{n_flag} of {len(r)} module{'' if len(r) == 1 else 's'} above the {threshold:.2f} threshold.")          # S-5
                        if preview:
                            base = rank(ranking.rename(columns={"module": "id"}), stored["ranking_key"])
                            was = set(base.loc[mark(base, stored["budget_unit"], stored["budget"] / 100,
                                                    stored["ranking_key"]).to_numpy().astype(bool), "id"])
                            now = set(r.loc[r["in_budget"].astype(bool), "id"])
                            st.caption("Compared with this build's settings: "                                  # S-9
                                       + ("no change to the in-budget selection." if was == now else
                                          f"{len(now - was)} module(s) added to and {len(was - now)} removed from the budget."))
                    sel = None
                    if len(r):
                        unmapped = set(((run.get("detail") or {}).get("stats") or {}).get("ranked_modules_missing_from_map", []))
                        r["tests mapped"] = ["no tests mapped" if m in unmapped else "yes" for m in r["id"]]
                        r["size"] = r["size"].round().astype("Int64")
                        # read-only states as text, not checkboxes (R-7); a flag is a badge (R-5)
                        show = pd.DataFrame({
                            "#": r["#"], "risk": r["p"],
                            "in budget": ["✓ in budget" if b else "–" for b in r["in_budget"].astype(bool)],
                            "flagged": ["⚑ above threshold" if f else "–" for f in r["flagged"]],
                            "tests mapped": r["tests mapped"], "module": r["id"], "lines of code": r["size"]})
                        if key == "effort":  # last, so the status columns stay in view; it orders, it does not flag (R-8)
                            show["ranking value: lower-bounded risk per 1,000 lines"] = \
                                (r["p"] * (1 - Z) + Z) / r["size"].where(r["size"] > 0) * 1000
                        shade = lambda row: ["background-color: rgba(31,58,95,0.10)" if row["in budget"] != "–" else ""
                                             for _ in row]
                        bold = lambda v: "font-weight: 600; color: #8a4b00" if v != "–" else ""
                        styled = show.style.apply(shade, axis=1).map(bold, subset=["flagged"]).format(
                            {"risk": "{:.2f}", "ranking value: lower-bounded risk per 1,000 lines": "{:.2f}"})
                        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)  # toolbar clears the line (R-13)
                        order = st.session_state.setdefault("rank-order", 0)
                        ev = st.dataframe(styled, on_select="rerun", selection_mode="single-row", hide_index=True,
                                          width="stretch", key=f"modules-{order}")
                        if st.button("Reset to rank order", help="Clears any column sorting and the row selection; "
                                                                  "# is always the rank (R-10, R-6)"):
                            st.session_state["rank-order"] = order + 1
                            st.rerun()
                        names = [f"{int(n)}. {m}" for n, m in zip(r["#"], r["id"])]
                        # the ticked row and the dropdown stay in step: ticking a row explains that row, and the
                        # dropdown (the keyboard path, R-3) can still choose any module (feedback 9 Nov)
                        pick_key = f"explain-{run_id}-{key}-{unit}"
                        row_now = ev.selection.rows[0] if ev.selection.rows else None
                        if row_now is not None and row_now != st.session_state.get(pick_key + "-row"):
                            st.session_state[pick_key] = names[row_now]
                        st.session_state[pick_key + "-row"] = row_now
                        if st.session_state.get(pick_key) not in names:
                            st.session_state[pick_key] = names[0]
                        sel = names.index(st.selectbox("Explain module (keyboard: Tab here, then arrow keys)", names,
                                                       key=pick_key))
                        if row_now is None and sel == 0:
                            st.caption("No row selected: the explanation shows rank 1 by default.")              # R-6
                        st.caption(f"Shaded rows = inside the budget. Flagged = calibrated risk above the {threshold:.2f} threshold"
                                   + (f"; lower-bounded risk = {1 - Z:.2f} × risk + {Z:.2f}." if key == "effort" else ".")
                                   + " Select a row to see its reasons.")
                    else:
                        st.info("This build touched no source module, so nothing was ranked. All tests ran in the trailing block.")
                with right:
                    if sel is not None:
                        row = r.iloc[sel]
                        explanation_panel(row["id"], row["p"], int(row["#"]), row["reason_codes"] or [], model,
                                          raw=row.get("raw"), domain=contribution_domain(r["reason_codes"]),
                                          default=not ev.selection.rows and sel == 0)

    with tab_plan, st.container(border=True):
            # ---------------------------------------------------------------- ordered test plan
            tp = db.test_plan(run_id)
            stats = (run.get("detail") or {}).get("stats") or {}
            st.markdown("## Ordered test plan: what the pipeline runs")
            st.caption("Built by the pipeline from the stored module ranking and settings"
                       + (", not from the preview above." if preview else "."))
            if tp.empty:
                st.info("No test plan stored for this build.")
            else:
                n_in = int(tp["in_budget"].sum())
                stored_rank = dict(zip(ranking["module"], ranking["rank"]))
                tp["from rank"] = tp["deciding_module"].map(stored_rank).astype("Int64")
                unit_word = "of the changed lines" if stored["budget_unit"] == "lines" else "of the changed modules"
                top_in = (int(ranking.loc[ranking["in_budget"].astype(bool), "rank"].max())
                          if "in_budget" in ranking and ranking["in_budget"].astype(bool).any() else 0)
                st.markdown(f"The {stored['budget']:g}% budget ({unit_word}) selects "
                            + ("no rank" if not top_in else "rank 1" if top_in == 1 else f"ranks 1–{top_in}")  # R-12
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
                    st.warning(f"The coverage map is old ({map_age(stats)}): tests for code added since then are not mapped.")
                # ranked and unranked tests apart (T-4); no "None" cells (T-5); status pinned on the left (T-6);
                # tables sized to their rows so the page, not the table, scrolls (T-7); wide test column (T-8)
                fit = lambda n: min(35 * (n + 1) + 3, 528)  # short tables in full; long ones capped at 14 rows
                cfg = {"#": st.column_config.NumberColumn(pinned=True, width="small"),
                       "in budget": st.column_config.TextColumn(pinned=True, width="small"),
                       "test": st.column_config.TextColumn(width="large")}
                ranked_tp = tp[tp["group"] == "ranked"]
                rest = tp[tp["group"] != "ranked"]
                if len(ranked_tp):
                    show = pd.DataFrame({"#": ranked_tp["position"],
                                         "in budget": ["✓ in budget" if b else "–" for b in ranked_tp["in_budget"]],
                                         "test": ranked_tp["test"],
                                         "module that sets its rank": ranked_tp["deciding_module"].fillna("—"),
                                         "from rank": ranked_tp["from rank"].map(lambda v: "—" if pd.isna(v) else int(v))})
                    styled = show.style.apply(lambda row: ["background-color: rgba(31,58,95,0.10)" if row["in budget"] != "–"
                                                           else "" for _ in row], axis=1)
                    st.markdown(f"**Ranked tests ({len(ranked_tp)})**, ordered by the rank of the modules they exercise")
                    st.dataframe(styled, hide_index=True, width="stretch", height=fit(len(show)), column_config=cfg)
                if len(rest):
                    why = {"no_ranked_module": "exercises no changed module", "no_coverage_record": "no coverage record",
                           "zero_size_only": "exercises only empty files", "default order": "default order"}
                    with st.expander(f"Unranked tests ({len(rest)}): run after the ranked tests, in test-name order"):
                        st.dataframe(pd.DataFrame({"#": rest["position"], "in budget": "–", "test": rest["test"],
                                                   "why unranked": rest["group"].map(lambda g: why.get(g, g))}),
                                     hide_index=True, width="stretch", height=fit(len(rest)), column_config=cfg)
                trailing = stats.get("trailing", {})
                missing = stats.get("ranked_modules_missing_from_map", [])
                zero = trailing.get("zero_size_only", 0)
                st.markdown(
                    f"Coverage map at commit `{str(run.get('coverage_map_commit') or 'unknown')[:7]}`"
                    + f" ({map_age(stats)})"
                    + f" · **{len(missing)}** ranked module{'' if len(missing) == 1 else 's'} not in the map"       # T-9
                    + (f" ({', '.join(missing)})" if missing else "")
                    + f" · of the {len(tp)} tests, {len(rest)} {'is' if len(rest) == 1 else 'are'} unranked: {trailing.get('no_ranked_module', 0)} exercise "
                    f"no changed module, {trailing.get('no_coverage_record', 0)} have no coverage record"
                    + (f", {zero} exercise only empty files, which are not ranked" if zero else ""))              # T-10

    with tab_off, st.container(border=True):
            # ---------------------------------------------------------------- evaluation panels
            off = db.latest_offline(module_model)
            st.markdown("## Offline model evaluation")
            st.info(f"These panels evaluate `{module_model}` offline on **{(off or {}).get('corpus') or 'its evaluation corpus'}**, "
                    f"not on `{project}`'s builds, and do not change with the settings above.")
            e1 = st.container()  # the cost-effectiveness curve on its own full-width row (walkthrough finding)
            e2, e3 = st.columns(2)
            with e1:
                st.markdown("### Cost-effectiveness (offline, lines-of-code budget)")
                if not off:
                    st.info(f"No offline evaluation is stored for `{module_model}` yet.")
                else:
                    st.caption(f"`{module_model}` · corpus: {off['corpus']} · evaluated {str(off['created'])[:10]} · "
                               "default key, independent of the settings above")
                    curves = (off.get("detail") or {}).get("curves", {})
                    d = pd.DataFrame([{"ordering": name, "lines inspected (%)": x * 100, "defect recall (%)": y * 100}
                                      for name, pts in curves.items() for x, y in pts])
                    if len(d):
                        # distinct hues, a legend that isolates a series on click (O-4), readable tooltips (O-5),
                        # axes that start at 0 (O-9) and a labelled budget line (O-8)
                        pick = alt.selection_point(fields=["ordering"], bind="legend")
                        line = alt.Chart(d).mark_line().encode(
                            x=alt.X("lines inspected (%):Q", scale=alt.Scale(domain=[0, 100], nice=False, padding=0)),
                            y=alt.Y("defect recall (%):Q", scale=alt.Scale(domain=[0, 100], nice=False, padding=0)),
                            strokeDash=alt.StrokeDash("ordering:N", sort=BASELINE_ORDER),
                            color=alt.Color("ordering:N", sort=BASELINE_ORDER, scale=alt.Scale(
                                domain=BASELINE_ORDER, range=["#1f3a5f", "#8c8c8c", "#b5834c", "#2a9d8f", "#6a4c93"])),
                            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.15)),
                            tooltip=[alt.Tooltip("ordering:N"), alt.Tooltip("lines inspected (%):Q", format=".1f"),
                                     alt.Tooltip("defect recall (%):Q", format=".1f")]).add_params(pick)
                        budget_line = pd.DataFrame({"x": [20], "label": ["20% budget"]})
                        rule = alt.Chart(budget_line).mark_rule(color="#b5834c").encode(
                            x=alt.X("x:Q", title="lines inspected (%)", scale=alt.Scale(domain=[0, 100], nice=False, padding=0)),
                            tooltip=[alt.Tooltip("label:N", title="line")])
                        label = alt.Chart(budget_line).mark_text(align="left", dx=4, dy=-6, color="#b5834c").encode(
                            x="x:Q", y=alt.value(8), text="label:N")
                        # theme=None: Streamlit's chart theme squeezed the gridlines inside the data range (0 and 100 drew
                        # off their gridlines; seen in Chrome, 9 Nov), so every chart here uses Vega-Lite's own theme
                        st.altair_chart((line + rule + label).configure_axis(offset=0), width="stretch", theme=None)
                        st.caption("Click a legend entry to isolate its curve; click empty space to show all.")
                    vals = [("system", off["recall"])] + [(b["baseline"], b["recall"]) for b in off["baselines"]
                                                           if b["recall"] is not None]
                    best = max(vals, key=lambda v: v[1])
                    st.markdown("Defect recall at the 20% budget: " + " · ".join(f"{n} {v:.1%}" for n, v in vals)
                                + f" (highest: {best[0]})")
            with e2:
                st.markdown("### Calibration: predicted against observed")
                bins = ((off or {}).get("detail") or {}).get("reliability", [])
                if not bins:
                    st.info("No reliability data stored for this model version yet.")
                else:
                    det = off["detail"]
                    prev = det.get("defective", 0) / det["rows"] if det.get("rows") else None
                    st.caption(f"Brier score: {det.get('brier', float('nan')):.3f}"                                # O-6
                               + (f" (a constant forecast of the defect rate, {prev:.1%}, scores {prev * (1 - prev):.3f}; "
                                  "lower is better)" if prev is not None else "")
                               + " · grey diagonal = perfect calibration · numbers = modules in each bin")
                    b = pd.DataFrame(bins)
                    prob_scale = alt.Scale(domain=[0, 1], nice=False, padding=0)
                    enc = dict(x=alt.X("mean_p:Q", title="predicted risk", scale=prob_scale),
                               y=alt.Y("observed:Q", title="observed defect rate", scale=prob_scale))
                    pts = alt.Chart(b).mark_line(point=True).encode(
                        **enc, tooltip=[alt.Tooltip("mean_p:Q", title="mean predicted risk", format=".3f"),
                                        alt.Tooltip("observed:Q", title="observed defect rate", format=".3f"),
                                        alt.Tooltip("n:Q", title="modules in bin")])                                  # O-5
                    counts = alt.Chart(b).mark_text(dy=-10, fontSize=11).encode(**enc, text="n:Q")                   # O-6
                    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="grey").encode(
                        x=alt.X("x:Q", scale=prob_scale), y=alt.Y("y:Q", scale=prob_scale))
                    st.altair_chart((diag + pts + counts).configure_axis(offset=0), width="stretch", theme=None)
            with e3:
                st.markdown("### Risk trend over recent builds")
                trend = db.risk_trend(project, limit=12)
                if trend.empty:
                    st.info("No build with in-budget modules yet.")
                else:
                    trend["build"] = trend["created"].astype(str).str[:16].str.replace("T", " ") + " " \
                        + trend["commit_after"].str[:7]
                    trend["selected"] = trend["id"] == run_id
                    st.caption(f"Average risk of in-budget modules, last {len(trend)} builds with a ranking"
                               + ("; the selected build is the larger dark point." if trend["selected"].any()
                                  else "; the selected build ranked no module or is older than these."))           # O-7
                    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)                      # O-10
                    x = alt.X("build:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45, labelLimit=140))      # O-7
                    tips = [alt.Tooltip("build:N"), alt.Tooltip("mean_in_budget_risk:Q", title="average risk", format=".3f"),
                            alt.Tooltip("in_budget_modules:Q", title="in-budget modules")]                            # O-5
                    base = alt.Chart(trend).encode(x=x, y=alt.Y("mean_in_budget_risk:Q", title="average risk"), tooltip=tips)
                    here = base.transform_filter("datum.selected").mark_point(size=160, filled=True, color="#1f3a5f")
                    st.altair_chart(base.mark_line(point=True, color="#8c8c8c") + here, width="stretch", theme=None)


if __name__ == "__main__":  # `streamlit run` executes the file as __main__; importing it does not
    main()
