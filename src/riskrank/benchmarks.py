"""The module-level benchmark corpora (section 3.5.1, Table 3.6): fetching with checksums, and loading.

    python -m riskrank.benchmarks fetch      # downloads into data/raw/, verifying every published checksum

NASA Metrics Data Program, cleaned (Shepperd et al., 2018; figshare 10.6084/m9.figshare.6071675, CC BY 4.0):
    the D'' versions (problem cases AND duplicates removed); KC4 is published empty, leaving 12 datasets. Duplicates are removed because a duplicate
    on both sides of a cross-validation split would leak a training row into the test fold. A module is
    a function; it is defective when its defect attribute is Y/true. Static metrics as supplied.

PROMISE, class level (Jureczko & Madeyski, 2010; the authors' Zenodo records, CC BY 4.0): every system
    with at least two releases (section 3.5.1), releases in version order. A module is a class; it is
    defective when its bug count is above zero. The 20 CK-family metrics as supplied; lines of code
    (loc) is the size for the lines budget.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

import pandas as pd

RAW = "data/raw"
NASA_ZIP = os.path.join(RAW, "nasa_mdp", "MDP.zip")
NASA_URL = "https://ndownloader.figshare.com/files/10937978"
NASA_MD5 = "f812f233529166c238cfb932334c97b4"
NASA_VERSION = "D''"

# system -> Zenodo record (Jureczko's uploads); systems with a single published release are omitted
PROMISE_RECORDS = {
    "camel": 268442, "ivy": 322436, "jedit": 268458, "log4j": 268451, "lucene": 322458, "poi": 322443,
    "prop": 322444, "synapse": 322449, "velocity": 322455, "xalan": 268436, "xerces": 268474,
}
PROMISE_DIR = os.path.join(RAW, "promise")
CK = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa", "mfa",
      "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"]


# ------------------------------------------------------------------ fetching
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "riskrank-benchmarks"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch() -> dict:
    manifest = {"nasa": {"url": NASA_URL, "md5": NASA_MD5}, "promise": {}}
    os.makedirs(os.path.dirname(NASA_ZIP), exist_ok=True)
    if not os.path.exists(NASA_ZIP):
        open(NASA_ZIP, "wb").write(_get(NASA_URL))
    assert hashlib.md5(open(NASA_ZIP, "rb").read()).hexdigest() == NASA_MD5, "NASA zip checksum mismatch"
    for system, rec in PROMISE_RECORDS.items():
        meta = json.loads(_get(f"https://zenodo.org/api/records/{rec}"))
        folder = os.path.join(PROMISE_DIR, system)
        os.makedirs(folder, exist_ok=True)
        files = {}
        for f in meta["files"]:
            if not f["key"].endswith(".csv"):
                continue
            path = os.path.join(folder, f["key"])
            if not os.path.exists(path):
                open(path, "wb").write(_get(f"https://zenodo.org/records/{rec}/files/{f['key']}?download=1"))
            algo, want = f["checksum"].split(":")
            got = hashlib.new(algo, open(path, "rb").read()).hexdigest()
            assert got == want, f"{path}: checksum mismatch"
            files[f["key"]] = f["checksum"]
        manifest["promise"][system] = {"zenodo_record": rec, "doi": meta["metadata"].get("doi"),
                                       "licence": meta["metadata"].get("license", {}).get("id"), "files": files}
    with open(os.path.join(RAW, "benchmarks_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


# ------------------------------------------------------------------ loading
def _arff(text: str) -> pd.DataFrame:
    names, rows, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if not in_data:
            if s.lower().startswith("@attribute"):
                names.append(re.split(r"\s+", s, maxsplit=2)[1].strip("'\""))
            elif s.lower().startswith("@data"):
                in_data = True
        else:
            rows.append([v.strip().strip("'\"") for v in s.split(",")])
    return pd.DataFrame(rows, columns=names)


def nasa() -> dict[str, pd.DataFrame]:
    """D'' datasets: numeric static metrics + `label` (bool) + `size` (LOC_TOTAL) + `id` (row number)."""
    out = {}
    with zipfile.ZipFile(NASA_ZIP) as z:
        for name in sorted(n for n in z.namelist() if f"/{NASA_VERSION}/" in n and n.endswith(".arff")):
            df = _arff(z.read(name).decode("latin-1"))
            if df.empty:  # KC4 is published with no data rows in both cleaned versions
                continue
            target = next(c for c in df.columns if c.lower() in ("defective", "label", "c", "problems"))
            y = df.pop(target).str.upper().isin(["Y", "TRUE", "YES"])
            x = df.apply(pd.to_numeric, errors="raise")
            x["label"] = y.to_numpy()
            x["size"] = x["LOC_TOTAL"]
            x["id"] = [f"{os.path.basename(name)[:-5]}:{i}" for i in range(len(x))]
            out[os.path.basename(name)[:-5]] = x
    return out


def _version_key(fname: str) -> tuple:
    v = fname.rsplit("-", 1)[1][:-4]
    return (-1,) if v == "init" else tuple(int(p) for p in v.split("."))


def promise() -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """system -> [(release, frame)] in version order; frame has the CK metrics, `label`, `size`, `id`."""
    out = {}
    for system in sorted(PROMISE_RECORDS):
        folder = os.path.join(PROMISE_DIR, system)
        releases = []
        for f in sorted(os.listdir(folder), key=_version_key):
            df = pd.read_csv(os.path.join(folder, f))
            df.columns = [c.strip().lower() for c in df.columns]
            x = df[CK].apply(pd.to_numeric, errors="raise")
            x["label"] = (pd.to_numeric(df["bug"]) > 0).to_numpy()
            x["size"] = x["loc"]
            # The files repeat the header "name" (system, then class); pandas reads the class column as
            # "name.1". The prop releases carry no class name, so their modules are numbered, as NASA's are.
            cls = df["name.1"].astype(str) if "name.1" in df.columns else pd.Series(range(len(df))).astype(str)
            x["id"] = (f[:-4] + ":" + cls).to_numpy()
            releases.append((f[:-4], x))
        if len(releases) >= 2:
            out[system] = releases
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        m = fetch()
        print(json.dumps({"nasa_md5": m["nasa"]["md5"], "promise": {k: len(v["files"]) for k, v in m["promise"].items()}}))
