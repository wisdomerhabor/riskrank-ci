"""Hook behaviour that does not need a running service."""
import subprocess

from riskrank.hook import module_features, summary


def test_push_touching_no_module_gives_empty_frame_with_columns(tmp_path):
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("x\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "c"], check=True)
    mf = module_features(str(repo), "HEAD", [], 12, r"\bfix\b")
    assert len(mf) == 0 and {"size", "complexity", "churn", "changes", "prior_fixes"} <= set(mf.columns)
    assert len(mf[mf["size"].isna()]) == 0  # the line that raised KeyError: 'size' in the live run


def test_summary_without_ranked_modules():
    plan = {"modules_touched": 0, "ranking": [], "escalated": False,
            "commit_level": {"status": "assessed", "commits": []},
            "test_plan": {"stats": {"in_budget_count": 0, "tests_total": 80, "in_budget_fraction": 0.0,
                                    "stale_map_warning": False}}}
    assert "**0** modules touched" in summary(plan)
