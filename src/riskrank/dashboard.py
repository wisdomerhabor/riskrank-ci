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
import re
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


def _md(v: str) -> str:
    """Escape a value and keep its **bold** and `code` (values sit inside an HTML block)."""
    v = html.escape(str(v), quote=False)
    v = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", v)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", v)


def facts(rows: list[tuple[str, str | None]], title: str | None = None,
          status: tuple[str, str] | None = None) -> None:
    """Labelled rows, one fact per line (feedback 9 Nov, replacing the '·' separator): a grey small-capitals
    label, the value beside it, important values in bold. Colour is used only for a status pill:
    status = ("warn" | "ok" | "info", text)."""
    head = ""
    if title:
        pill = f' <span class="rr-pill rr-{status[0]}">{html.escape(status[1])}</span>' if status else ""
        head = f'<div class="rr-t">{html.escape(title)}{pill}</div>'
    body = "".join(f'<div class="rr-k">{html.escape(k)}</div><div class="rr-c">:</div><div class="rr-v">{_md(v)}</div>'
                   for k, v in rows if v not in (None, ""))
    st.markdown(f'<div class="rr-facts">{head}<div class="rr-grid">{body}</div></div>', unsafe_allow_html=True)


def escalation_facts(run: dict, commits: pd.DataFrame) -> tuple[tuple[str, str], list[tuple[str, str]]]:
    """The pipeline's escalation decision as (status pill, rows), the deciding value unrounded beside the
    threshold (HD-1, HD-2, HD-10). The settings bar does not change it."""
    status = run["jit_status"]
    if status == "not configured":
        return ("info", "Escalation not configured"), [("Escalation", "no just-in-time model is configured")]
    if status != "assessed":
        return ("info", "Not assessed"), [("Escalation", f"not assessed ({status})")]
    inside = commits[commits["in_training_range"].astype(bool)] if len(commits) else commits
    thr = float(run["risk_threshold"] or 0.5)
    top = inside["p"].max() if len(inside) else None
    yes = bool(run["escalated"])
    value = ("**Yes**" if yes else "**No**") + ("" if top is None else
             f" (highest commit-level risk **{top:.4f}** {'exceeds' if top > thr else 'does not exceed'} "
             f"threshold **{thr:.2f}**)")
    return (("warn", "Escalated") if yes else ("ok", "Not escalated")), [
        ("Escalation", value),
        ("Decided by", "the pipeline, using the threshold in `riskrank.yml`"),
        ("Counts", "only commits whose features lie inside the model's training range")]


# one status vocabulary for every table (feedback 9 Nov): budget and risk stay separate signals, and a module
# below the threshold is "below threshold", not "safe", so it gets no green
STATUS_CSS = {
    "In budget": "background-color: #e0f2e6; color: #1e6b3a; font-weight: 600",
    "Flagged": "background-color: #fdecd2; color: #8a4b00; font-weight: 600",
    "Unmapped": "background-color: #fdecd2; color: #8a4b00",
    "Outside training range": "background-color: #fdecd2; color: #8a4b00",
    "Mapped": "color: #1e6b3a",
    "Outside budget": "color: #6b7280", "Below threshold": "color: #6b7280", "Inside training range": "color: #6b7280",
}


def status_css(v) -> str:
    return STATUS_CSS.get(v, "")


def risk_column(digits: int = 2, width="small") -> st.column_config.NumberColumn:
    """The calibrated risk as a plain number (the progress bar was removed at the student's request, 9 Nov)."""
    return st.column_config.NumberColumn("Risk score", format=f"%.{digits}f", width=width,
                                         help="Calibrated probability that the module or commit is defect-prone, "
                                              "on a 0–1 scale")


def build_label(r) -> str:
    """A build option with its status, so an escalated or degraded build is found without opening each
    one (walkthrough HD-5)."""
    marks = [m for m, on in (("escalated", r.escalated), ("default order", r.status == "default order"),
                             ("partial", r.status == "partial"), ("stale map", r.stale_map)) if on]
    return (f"{str(r.created)[:16].replace('T', ' ')}  |  {str(r.commit_after or '')[:7]}  |  run {r.build_ref}"
            + (f"  |  {', '.join(marks)}" if marks else ""))


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
    x = alt.X("contribution:Q", title="contribution to the uncalibrated score", axis=alt.Axis(tickCount=8),
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
    # a long path wraps at any point instead of overflowing (.rr-v code); copying it gives the plain path (E-8)
    facts([("Selected", f"`{title}`" + (" (shown by default; select a row to change)" if default else "")),  # E-6, E-8
           ("Risk", f"**{p:.{digits}f}**"), ("Rank", str(rank_no) if rank_no else None)])
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
                ".rr-facts {margin: 0.15rem 0 0.9rem} "
                ".rr-t {font-weight: 600; font-size: 0.98rem; margin-bottom: 0.45rem} "
                ".rr-grid {display: grid; grid-template-columns: max-content auto 1fr; column-gap: 0.6rem; "
                "row-gap: 0.32rem; align-items: baseline} "
                ".rr-k {font-size: 0.9rem; color: #6b7280; white-space: nowrap} "
                ".rr-c {color: #9ca3af} "
                ".rr-v {font-size: 0.95rem; line-height: 1.45} "
                ".rr-v code {font-size: 0.85rem; white-space: normal; overflow-wrap: anywhere} "
                ".rr-pill {display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 0.08rem 0.6rem; "
                "border-radius: 999px; margin-left: 0.5rem; vertical-align: 0.1rem} "
                ".rr-warn {background: #fdecd2; color: #8a4b00} .rr-ok {background: #e0f2e6; color: #1e6b3a} "
                ".rr-info {background: #eef1f5; color: #44505f} "
                "[data-testid='stColumn']:has(.st-key-explain-pane) {position: sticky; top: 4.5rem; align-self: flex-start; "
                "z-index: 1}"
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
        pill, esc_rows = escalation_facts(run, commits_this)
        # the status tiles live here now (feedback 9 Nov); "Build escalated" was a third copy of the pill and
        # the Escalation row, so it went rather than moved
        facts([("Pushed commit", f"`{str(run['commit_after'] or '')[:7] or '–'}`"),
               (f"{'Commit' if view_is_commits else 'Module'}-level model", f"`{shown_model}` (calibrated)"),
               ("Pipeline status", str(run["status"] or "unknown"))]
              + esc_rows
              + [("Coverage map age", map_age(((run.get("detail") or {}).get("stats") or {}))),
                 ("Signed in", f"**{user}**")],
              title="Build decision", status=pill)
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

    # status (L-3, L-4) is the Build decision block above the tabs: a pill plus labelled rows
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
            left, right_col = st.columns([2, 1])
            # the explanation sits beside the table and stays in view while the page scrolls (feedback 9 Nov)
            right = right_col.container(key="explain-pane")
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
                            view = shown.assign(
                                commit=shown["commit"].str[:10], risk=shown["p"],
                                flagged=["Flagged" if v > threshold else "Below threshold" for v in shown["p"]],
                                features_outside_training_range=["Inside training range" if i else "Outside training range"
                                                                 for i in shown["in_training_range"].astype(bool)])
                            view["created"] = pd.to_datetime(view["created"], utc=True).dt.strftime("%Y-%m-%d %H:%M")  # C-6
                            table = view[["commit", "risk", "flagged", "features_outside_training_range", "created",
                                          "model_version"]].rename(columns={
                                              "commit": "Commit", "risk": "Risk score", "flagged": "Flagged",
                                              "created": "Scored in build at", "model_version": "Model",
                                              "features_outside_training_range": "Features"})
                            ev = st.dataframe(table.style.map(status_css, subset=["Flagged", "Features"]),
                                              on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch",
                                              key="commits", column_config={"Model": st.column_config.TextColumn(width="medium"),
                                                                            "Risk score": risk_column(4)})
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
                    over = share is not None and unit == "lines" and share > budget / 100 + 1e-9
                    facts([("Touched", f"**{touched}** module{'' if touched == 1 else 's'}"),
                           ("Budget", f"**{budget:g}%** {'of changed lines' if unit == 'lines' else 'of changed modules'}"
                                      + (", covering " + ("rank 1" if last_in == 1 else f"ranks 1–{last_in}")
                                         if k else ", covering no module")),
                           ("Realised share", None if share is None else f"**{share:.1%}**"
                            + (" (the module at the boundary is included whole, so the share can exceed the budget)"
                               if over else "")),
                           ("Flagged", None if not len(r) else                                                    # S-5
                            f"**{int(r['flagged'].sum())}** of {len(r)} module{'' if len(r) == 1 else 's'} above "
                            f"the {threshold:.2f} threshold")])
                    if len(r):
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
                        r["tests mapped"] = ["Unmapped" if m in unmapped else "Mapped" for m in r["id"]]
                        r["size"] = r["size"].round().astype("Int64")
                        # read-only states as text, not checkboxes (R-7); a flag is a badge (R-5)
                        effort_col = "Ranking value (lower-bounded risk per 1,000 LOC)"
                        show = pd.DataFrame({
                            "#": r["#"], "Risk score": r["p"],
                            "Budget": ["In budget" if b else "Outside budget" for b in r["in_budget"].astype(bool)],
                            "Flagged": ["Flagged" if f else "Below threshold" for f in r["flagged"]],
                            "Tests mapped": r["tests mapped"], "Module": r["id"], "LOC": r["size"]})
                        if key == "effort":  # last, so the status columns stay in view; it orders, it does not flag (R-8)
                            show[effort_col] = (r["p"] * (1 - Z) + Z) / r["size"].where(r["size"] > 0) * 1000
                        styled = show.style.map(status_css, subset=["Budget", "Flagged", "Tests mapped"]).format(
                            {effort_col: "{:.2f}"} if key == "effort" else {})
                        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)  # toolbar clears the line (R-13)
                        order = st.session_state.setdefault("rank-order", 0)
                        ev = st.dataframe(styled, on_select="rerun", selection_mode="single-row", hide_index=True,
                                          width="stretch", key=f"modules-{order}", column_config={
                                              "Risk score": risk_column(width="small"),
                                              "LOC": st.column_config.NumberColumn(format="%d", help="Lines of code")})
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
                        st.caption(f"Budget = inside or outside the testing budget. Flagged = calibrated risk above the "
                                   f"{threshold:.2f} threshold. Unmapped = no test in the coverage map exercises the module"
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
                facts([("Budget", f"**{stored['budget']:g}%** {unit_word}, selecting "
                                  + ("no rank" if not top_in else "rank 1" if top_in == 1 else f"ranks 1–{top_in}")),  # R-12
                       ("In-budget tests", f"**{n_in}** of {len(tp)} ({n_in / len(tp):.1%} of the suite)")])
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
                       "Budget": st.column_config.TextColumn(pinned=True, width=115),
                       "Test": st.column_config.TextColumn(width="large")}
                ranked_tp = tp[tp["group"] == "ranked"]
                rest = tp[tp["group"] != "ranked"]
                if len(ranked_tp):
                    show = pd.DataFrame({"#": ranked_tp["position"],
                                         "Budget": ["In budget" if b else "Outside budget" for b in ranked_tp["in_budget"]],
                                         "Test": ranked_tp["test"],
                                         "Module that sets its rank": ranked_tp["deciding_module"].fillna("—"),
                                         "From rank": ranked_tp["from rank"].map(lambda v: "—" if pd.isna(v) else int(v))})
                    styled = show.style.map(status_css, subset=["Budget"])
                    st.markdown(f"**Ranked tests ({len(ranked_tp)})**, ordered by the rank of the modules they exercise")
                    st.dataframe(styled, hide_index=True, width="stretch", height=fit(len(show)), column_config=cfg)
                if len(rest):
                    why = {"no_ranked_module": "exercises no changed module", "no_coverage_record": "no coverage record",
                           "zero_size_only": "exercises only empty files", "default order": "default order"}
                    with st.expander(f"Unranked tests ({len(rest)}): run after the ranked tests, in test-name order"):
                        st.dataframe(pd.DataFrame({"#": rest["position"], "Budget": "Outside budget", "Test": rest["test"],
                                                   "Why unranked": rest["group"].map(lambda g: why.get(g, g))}),
                                     hide_index=True, width="stretch", height=fit(len(rest)), column_config=cfg)
                trailing = stats.get("trailing", {})
                missing = stats.get("ranked_modules_missing_from_map", [])
                zero = trailing.get("zero_size_only", 0)
                facts([("Map commit", f"`{str(run.get('coverage_map_commit') or 'unknown')[:7]}`"), ("Map age", map_age(stats)),
                       ("Not in the map", f"**{len(missing)}** ranked module{'' if len(missing) == 1 else 's'}"  # T-9
                                          + (f" ({', '.join(missing)})" if missing else "")),
                       ("Unranked tests", f"**{len(rest)}** of the {len(tp)}: {trailing.get('no_ranked_module', 0)} "
                                          f"exercise no changed module, {trailing.get('no_coverage_record', 0)} have "
                                          "no coverage record"
                                          + (f", {zero} exercise only empty files, which are not ranked" if zero else ""))],  # T-10
                      title="Plan details")

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
                    facts([("Model", f"`{module_model}`"), ("Corpus", off["corpus"]),
                           ("Evaluated", str(off["created"])[:10]),
                           ("Ranking key", "default, independent of the settings above")])
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
                    facts([(n, f"**{v:.1%}**" + ("  (highest)" if n == best[0] else ""))
                           for n, v in sorted(vals, key=lambda v: -v[1])], title="Defect recall at the 20% budget")
            with e2:
                st.markdown("### Calibration: predicted against observed")
                bins = ((off or {}).get("detail") or {}).get("reliability", [])
                if not bins:
                    st.info("No reliability data stored for this model version yet.")
                else:
                    det = off["detail"]
                    prev = det.get("defective", 0) / det["rows"] if det.get("rows") else None
                    facts([("Brier score", f"**{det.get('brier', float('nan')):.3f}** (lower is better)"),       # O-6
                           ("Baseline", None if prev is None else
                            f"a constant forecast of the defect rate, {prev:.1%}, scores {prev * (1 - prev):.3f}"),
                           ("Reading", "grey diagonal = perfect calibration; numbers = modules in each bin")])
                    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)  # toolbar clears the rows
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
