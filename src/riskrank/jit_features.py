"""The shared just-in-time feature implementation (Table 3.7, NFR12).

The 12 change metrics of Kamei et al. (2013), computed from a git repository for a given commit.
The ApacheJIT replication package ships these metrics precomputed and contains no code that computes
them, so the definitions below were fixed empirically: each variant was recomputed on 60 ZooKeeper
commits and compared with the published values (`python -m riskrank.nfr12`). Chosen definitions:

  la, ld        lines added / deleted over ALL files the commit changes (not only .java)
  nf, nd, ns    distinct files, directories, and root directories (subsystems)
  ent           -sum p_k log2 p_k, p_k = file k's changed lines / all changed lines (unnormalised)
  ndev          mean over the changed files of the distinct prior authors of each file
  age           mean over the changed files of days since each file's previous change (0 if new)
  nuc           mean over the changed files of the number of prior commits that changed each file
  aexp          the author's prior commits
  arexp         sum over the author's prior commits of 1 / (1 + whole years since that commit)
  asexp         the author's prior commits touching any subsystem this commit touches
History is the repository's main-branch history (no merges) strictly before the commit's author
time; authors are identified by lower-cased e-mail address.
"""

from __future__ import annotations

import math
import subprocess
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass

FEATURES = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
YEAR = 365 * 86400


@dataclass
class Commit:
    sha: str
    author: str
    t: int
    files: list[tuple[str, int, int]]


class History:
    """Main-branch, non-merge history of one repository, indexed for fast per-commit queries."""

    def __init__(self, repo: str, rev: str = "HEAD"):
        out = subprocess.run(
            ["git", "-C", repo, "log", rev, "--no-merges", "--no-renames", "--numstat",
             "--format=\x1e%H%x00%ae%x00%at"],
            capture_output=True, text=True, check=True).stdout
        commits = []
        for rec in out.split("\x1e")[1:]:
            head, _, stats = rec.partition("\n")
            h, ae, at = head.split("\x00")
            files = []
            for line in stats.strip().splitlines():
                a, d, p = line.split("\t", 2)
                files.append((p, 0 if a == "-" else int(a), 0 if d == "-" else int(d)))
            commits.append(Commit(h, ae.lower(), int(at), files))
        commits.sort(key=lambda c: c.t)
        self.commits = commits
        self.by_sha = {c.sha: c for c in commits}
        self.times = [c.t for c in commits]
        self.file_hist: dict[str, list[Commit]] = defaultdict(list)
        self.author_hist: dict[str, list[Commit]] = defaultdict(list)
        for c in commits:
            self.author_hist[c.author].append(c)
            for p, _, _ in c.files:
                self.file_hist[p].append(c)

    def features(self, sha: str) -> dict[str, float]:
        c = self.by_sha[sha]
        return compute(c, self)


def _before(seq: list[Commit], t: int) -> list[Commit]:
    return seq[: bisect_left([x.t for x in seq], t)]


def compute(c: Commit, h: History) -> dict[str, float]:
    files = c.files
    paths = sorted({p for p, _, _ in files})
    la, ld = sum(a for _, a, _ in files), sum(d for _, _, d in files)
    tot = la + ld
    ent = -sum(((a + d) / tot) * math.log2((a + d) / tot) for _, a, d in files if a + d) if tot else 0.0
    subsystems = {p.split("/", 1)[0] for p in paths}
    devs, nuc, age = [], [], []
    for p in paths:
        prior = _before(h.file_hist.get(p, []), c.t)
        devs.append(len({x.author for x in prior}))
        nuc.append(len(prior))
        age.append((c.t - prior[-1].t) / 86400 if prior else 0.0)
    mine = _before(h.author_hist.get(c.author, []), c.t)
    n = max(1, len(paths))
    return {
        "la": la, "ld": ld, "nf": len(paths),
        "nd": len({p.rsplit("/", 1)[0] if "/" in p else "" for p in paths}),
        "ns": len(subsystems), "ent": ent,
        "ndev": sum(devs) / n, "age": sum(age) / n, "nuc": sum(nuc) / n,
        "aexp": len(mine),
        "arexp": sum(1 / (1 + (c.t - x.t) // YEAR) for x in mine),
        "asexp": sum(1 for x in mine if subsystems & {p.split("/", 1)[0] for p, _, _ in x.files}),
    }
