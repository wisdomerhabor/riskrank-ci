"""R-SZZ with file attribution, run inside the pyszz:2026 image. PySZZ itself is not modified.

    docker run --rm -v "$PWD/data:/data" -v "$PWD/tools/pyszz:/drv" --entrypoint python3 \
        pyszz:2026 /drv/drive_rszz.py /data/mined/flask/bugfixes.json /data/repos /data/mined/flask/rszz.jsonl

For each fix, mirroring PySZZ main.py's R-SZZ call argument for argument:
  1. get_impacted_files(fix, file_ext_to_parse=[py], only_deleted_lines=True), kept to the fix's
     pre-registered source files.
  2. Stock R-SZZ on that list -> the one inducing commit R-SZZ picks.
  3. MA-SZZ on each file alone -> that file's candidates. R-SZZ's rule (latest by committed date)
     over their union must give the same commit as step 2; this is asserted for every fix.
  4. Blamed modules = the files whose own candidates include that commit.
The issue-date filter is switched on per fix, only where the fix carries an issue date: published
main.py asserts that every fix has one whenever the filter is enabled.
"""

import json
import sys
import time

import yaml

sys.path.insert(0, "/usr/src/app")
from szz.common.issue_date import parse_issue_date  # noqa: E402
from szz.ma_szz import MASZZ  # noqa: E402
from szz.r_szz import RSZZ  # noqa: E402
from szz.core.abstract_szz import DetectLineMoved  # noqa: E402

CONF = yaml.safe_load(open("/usr/src/app/conf/rszz_issues_filter.yml"))


def kwargs_for(fix):
    has_date = "earliest_issue_date" in fix
    return dict(
        max_change_size=CONF.get("max_change_size"),
        detect_move_from_other_files=DetectLineMoved(CONF.get("detect_move_from_other_files")),
        issue_date_filter=has_date,
        issue_date=parse_issue_date(fix) if has_date else None,
        filter_revert_commits=CONF.get("filter_revert_commits", False),
    )


def main(fixes_path, repos_dir, out_path):
    fixes = json.load(open(fixes_path))
    done = set()
    try:
        done = {json.loads(l)["fix_commit_hash"] for l in open(out_path)}
    except FileNotFoundError:
        pass
    out = open(out_path, "a")
    for i, fix in enumerate(fixes):
        h = fix["fix_commit_hash"]
        if h in done:
            continue
        t0 = time.time()
        repo = fix["repo_name"]
        url = f"https://github.com/{repo}.git"
        r = RSZZ(repo_full_name=repo, repo_url=url, repos_dir=repos_dir)
        imp = r.get_impacted_files(fix_commit_hash=h, file_ext_to_parse=CONF.get("file_ext_to_parse") or ["py"],
                                   only_deleted_lines=True)
        imp = [f for f in imp if f.file_path in set(fix["source_files"])]
        kw = kwargs_for(fix)
        rec = {"fix_commit_hash": h, "fix_date": fix["fix_date"], "impacted": [f.file_path for f in imp],
               "issue_date_filter": kw["issue_date_filter"]}
        if not imp:
            rec |= {"inducing": None, "blamed_files": [], "note": "no deleted source lines to trace"}
        else:
            stock = r.find_bic(fix_commit_hash=h, impacted_files=imp, **kw)
            stock = next(iter(stock))
            per_file, union = {}, set()
            for f in imp:
                m = MASZZ(repo_full_name=repo, repo_url=url, repos_dir=repos_dir)
                cands = m.find_bic(fix_commit_hash=h, impacted_files=[f], **kw)
                per_file[f.file_path] = sorted(c.hexsha for c in cands)
                union |= cands
            mine = RSZZ.select_latest_commit(union)
            stock_h = stock.hexsha if stock is not None else None
            mine_h = mine.hexsha if mine is not None else None
            assert stock_h == mine_h, f"file attribution diverged from stock R-SZZ on {h}: {stock_h} vs {mine_h}"
            rec |= {"inducing": stock_h,
                    # read through `stock`: its R-SZZ object (and repo copy) is still alive here,
                    # whereas each per-file MA-SZZ object deletes its temporary repo copy on cleanup
                    "inducing_date": stock.committed_datetime.isoformat() if stock is not None else None,
                    "candidates_per_file": per_file,
                    "blamed_files": [p for p, c in per_file.items() if stock_h in c]}
        rec["seconds"] = round(time.time() - t0, 1)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{i + 1}/{len(fixes)}] {h[:8]} -> {str(rec['inducing'])[:8]} files={len(rec['blamed_files'])} "
              f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
