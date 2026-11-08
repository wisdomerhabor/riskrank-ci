# riskrank-ci

Ranks the source files a push touches by predicted defect risk, then orders the repository's own
tests so that tests for the riskiest files run first. Runs inside GitHub Actions; never fails a build.

Prototype for an MIT (Software Engineering) project at Miva Open University.

## What runs on each push

1. `riskrank.hook` reads `riskrank.yml`, finds the touched modules and the push's commits, and computes
   their features from git history.
2. A scoring container (job service) returns a calibrated risk and SHAP reason codes for each one.
3. Modules are ranked, a testing budget (share of files or of lines) is applied, and a per-test coverage
   map turns the ranking into an ordered test plan. Every test stays in the plan.
4. Tests run in that order; the plan and a short risk summary are attached to the run.

If anything fails (bad settings, model checksum, scoring timeout, missing coverage map) the job logs a
warning and runs the tests in their usual order.

## Workstation side: plan store, plan ingest and dashboard

The CI runner cannot reach the workstation, so the workstation pulls. `riskrank.ingest` collects each
completed scoring run's plan artifact from the default branch, checks where it came from and its format,
and saves it through the storage interface (`riskrank.store`, SQLite by default, PostgreSQL by URL).
The Streamlit dashboard (`src/riskrank/dashboard.py`) reads only from that store and asks for a login
before showing anything.

```bash
.venv/bin/pip install -e ".[dashboard]"
.venv/bin/python -m riskrank.auth ~/.config/riskrank/dashboard-auth.json
RISKRANK_GH_TOKEN_FILE=~/.config/riskrank/ingest-token .venv/bin/python -m riskrank.ingest --repo wisdomerhabor/riskrank-ci
RISKRANK_AUTH=~/.config/riskrank/dashboard-auth.json .venv/bin/streamlit run src/riskrank/dashboard.py --server.address 127.0.0.1
```

Or run both as containers with `docker compose -f deploy/compose.yaml up -d --build`.

## Layout

| Path | Contents |
|---|---|
| `src/riskrank/` | data preparation, models, calibration, evaluation, service, hook |
| `ci/github/` | workflow templates to copy into a repository that uses the tool |
| `config/flask_mining.yaml` | the fixed rules used to label the Flask history |
| `tools/pyszz/`, `tools/radon27/` | container images for the SZZ labelling and Python 2 metrics |
| `tests/` | unit and service tests (`pytest`) |

## Try it locally

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e . pytest
.venv/bin/pytest -q
docker build -t riskrank-scoring .
```

Datasets (ApacheJIT from Zenodo record 5907847, a clone of pallets/flask) and trained models are not
stored in this repository.

## Licence notes

PySZZ v2 (GPL-3.0) is run as a separate tool inside its own container and is not imported.
