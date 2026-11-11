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

## Privacy

The project processes developer data from public commit histories under the Nigeria Data Protection
Act 2023. See [PRIVACY.md](PRIVACY.md) for the privacy notice, including how to object.

## Reproduce the project

This section lets an examiner inspect and rerun the implementation described in the project report.
Commands run from the repository root unless stated otherwise. Inputs that are not in git go under
`data/` and outputs under `runs/`; both folders are ignored by git.

### Versions used in the report

| Item | Version | Where |
|---|---|---|
| Code | release `v1.0.0` | this repository |
| Scoring image | `ghcr.io/wisdomerhabor/riskrank-scoring:v0.1.8` | built from tag `v0.1.8` by `.github/workflows/publish-image.yml` |
| Model bundles | `flask-module-v3`, `apachejit-jit-v1` | assets of release `v0.1.3` (`models-*.tar.gz`) |
| Model checksums | SHA-256 of each `bundle.joblib` | `riskrank.yml`, `models:` |
| Second deployment | fork `wisdomerhabor/flask` | its `main` holds the 34-batch `flask-module-v3` replay |
| Flask history mined | `pallets/flask` at `49943b5a94413a530761350f04e896f76e8d620e` | `config/flask_mining.yaml`, `history_end_commit` |

Both workflows (`.github/workflows/riskrank.yml` here and the fork's) set `RISKRANK_REF: v0.1.8` and
`MODELS_REF: v0.1.3`.

### Install

Python 3.13 is required (`requires-python = ">=3.13"`). The pinned set below is the one the
evaluation workflows install.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt scipy==1.18.1 threadpoolctl==3.8.0 joblib==1.6.0
.venv/bin/pip install -e ".[service,dashboard]" pytest-cov
```

Extras in `pyproject.toml`: `ml` (model libraries), `service` (`ml` plus Flask and gunicorn), `dashboard`
(Streamlit, Altair, SQLAlchemy) and `postgres`. The base install without extras is what the CI hook
needs inside another project.

### Run the test suite

```bash
.venv/bin/pytest -q
```

The suite collects 197 tests. With all datasets downloaded the result is 194 passed and 3 skipped.
On a fresh clone, the ApacheJIT and benchmark tests are also skipped until their data is downloaded
(see Data below).

### Run the scoring service

The service needs the model bundles and a `models.json` that names them with their checksums.

```bash
mkdir -p models
gh release download v0.1.3 -R wisdomerhabor/riskrank-ci -p 'models-*.tar.gz' -D models
for f in models/models-*.tar.gz; do tar xzf "$f" -C models; done
.venv/bin/python -m riskrank.models_json riskrank.yml > models/models.json
```

With the published image:

```bash
docker run --rm -p 8000:8000 -v "$PWD/models:/models:ro" ghcr.io/wisdomerhabor/riskrank-scoring:v0.1.8
curl http://localhost:8000/health
```

Or build the image yourself with `docker build -t riskrank-scoring .` and run it the same way. Without
Docker, run it from the virtual environment (the folders in `models.json` must then be host paths):

```bash
.venv/bin/python -m riskrank.models_json riskrank.yml "$PWD/models" > models/models-host.json
RISKRANK_MODELS=models/models-host.json .venv/bin/flask --app riskrank.service run --port 8000
```

`GET /health` lists the loaded model versions, their checksums and input features. `POST /score` is
described in `src/riskrank/service.py`.

### Build the coverage map

The per-test coverage map comes from one full suite run with per-test contexts. This is the step in
`.github/workflows/coverage.yml`, which publishes the result as the `coverage-latest` release asset.

```bash
RISKRANK_DURATIONS=durations.json .venv/bin/python -m pytest -q -p riskrank.durations_plugin \
    --cov=. --cov-context=test --cov-report=
.venv/bin/python -m riskrank.build_map --data .coverage --repo . --durations durations.json --out coverage-map.json
```

### Run the CI hook locally

With the scoring service running on port 8000 and a coverage map (built above, or downloaded with
`gh release download coverage-latest -R wisdomerhabor/riskrank-ci -p coverage-map.json`):

```bash
RISKRANK_COLLECT=tests.jsonl .venv/bin/python -m pytest --collect-only -q -p riskrank.order_plugin > /dev/null
.venv/bin/python -m riskrank.hook --repo . --before HEAD~1 --after HEAD --settings riskrank.yml \
    --service http://localhost:8000 --coverage coverage-map.json --tests tests.jsonl --out riskrank-plan
.venv/bin/python -m riskrank.run_tests riskrank-plan riskrank.yml
```

`riskrank-plan/` then holds `plan.json`, `order.txt`, `in_budget.txt` and `summary.md`. If the service
is not running, the hook warns and writes a plan in the default test order; it never fails.

### Add RiskRank to another repository

1. Copy `ci/github/riskrank.yml` and `ci/github/coverage.yml` into the repository's
   `.github/workflows/`. Set `branches:` in `riskrank.yml` to its default branch.
2. Add a `riskrank.yml` settings file at the repository root. The fields are documented in
   `src/riskrank/settings.py`; this repository's `riskrank.yml` and the fork's are worked examples.
   Set `source_roots` and `exclude_paths` for the project's layout. Leave `models.commit` out, and
   `deploying_team` empty, if commits should not be scored.
3. Run the `riskrank-coverage` workflow once by hand (`workflow_dispatch`) so the first coverage map
   exists. After that it runs weekly.
4. Push to the default branch. Each push produces a `riskrank-plan` artifact, kept for 7 days.

The templates install RiskRank at `RISKRANK_REF` and load the models from release `MODELS_REF` of
`wisdomerhabor/riskrank-ci`.

### Data

**ApacheJIT** (Zenodo record 5907847). The code expects the archive at `data/raw/apachejit.zip`,
unpacked to `data/raw/apachejit/`. The run manifests record its SHA-256
(`533e7c6fc4f7cb6f34b35539a972d00baa63a125e88ddb5743283dd277edcd5c`).

```bash
mkdir -p data/raw
curl -L -o data/raw/apachejit.zip "https://zenodo.org/records/5907847/files/apachejit_dataset_replication.zip?download=1"
unzip -q data/raw/apachejit.zip -d data/raw
```

**NASA MDP (cleaned, D'') and PROMISE.** One command downloads both into `data/raw/` and checks every
published checksum: NASA MDP from figshare file 10937978, and PROMISE from the authors' Zenodo records
listed in `src/riskrank/benchmarks.py`.

```bash
.venv/bin/python -m riskrank.benchmarks fetch
```

**Flask corpus.** Mining and labelling follow the rules fixed in `config/flask_mining.yaml`. They need a
clone of Flask, Docker, and two tool images:

```bash
git clone https://github.com/pallets/flask data/repos/pallets/flask
sh tools/pyszz/validate.sh                       # builds pyszz:2026 and runs PySZZ's own R-SZZ test
docker build -t radon27:4.1.0 tools/radon27      # Radon 4.1.0 for Python 2-only source
```

The PySZZ image is built for arm64 (Apple silicon): `tools/pyszz/Dockerfile` downloads the arm64
srcML package.

```bash
.venv/bin/python -m riskrank.mining.fixes        # -> data/mined/flask/bugfixes.json
docker run --rm -v "$PWD/data:/data" -v "$PWD/tools/pyszz:/drv" --entrypoint python3 \
    pyszz:2026 /drv/drive_rszz.py /data/mined/flask/bugfixes.json /data/repos /data/mined/flask/rszz.jsonl
.venv/bin/python -m riskrank.mining.snapshots    # -> data/mined/flask/modules.csv (module level)
.venv/bin/python -m riskrank.mining.commits      # -> data/mined/flask/commits.csv (commit level)
.venv/bin/python -m riskrank.leakage_check       # -> runs/leakage-check.json
```

`riskrank.mining.fixes` reads a cache of Flask issue labels and dates at `data/github/flask_issues.json`.
That cache is not published and no script in this repository fetches it. To label without it, use the
published bug-fix list: `bugfixes.json` inside `second-check-inputs.tar.gz` on release
`eval-inputs-20261109`. Put it at `data/mined/flask/bugfixes.json` and start from the R-SZZ step.

**What is published and what is not.** The module-level snapshot corpus (`modules.csv`) is attached to
release `v1.0.0` as `flask-module-snapshots.zip`. The commit-level corpus (`commits.csv`) is **not**
published. It holds developer-experience metrics, which are personal data under the Nigeria Data
Protection Act 2023 (see [PRIVACY.md](PRIVACY.md)). It can be rebuilt from public sources with the
commands above. Rebuilt ApacheJIT partitions are not published either.

### Offline evaluations

Model fitting is time-budgeted (`--seconds`), so the number of tuning trials depends on the machine. The
run records hold the tuned hyperparameters, which is what makes a run repeatable. Run these on an idle
machine.

ApacheJIT, within-project and cross-project (the within-project run of record is
`runs/within-20261108T224535Z`):

```bash
.venv/bin/python -m riskrank.run_within --seconds 60 --seed 42     # -> runs/within-<time>/
.venv/bin/python -m riskrank.evaluate runs/within-<time>            # -> evaluation.json
.venv/bin/python -m riskrank.sk_esd runs/within-<time>/evaluation.json
.venv/bin/python -m riskrank.run_cross --seconds 60 --seed 42      # -> runs/cross-<time>/
.venv/bin/python -m riskrank.evaluate runs/cross-<time>
```

Sensitivity variants use `run_within --variant` (`exclude_relabelled`, `window_1y`, `from_2010`).

Deployed models:

```bash
.venv/bin/python -m riskrank.deploy_jit --seconds 120 --seed 42 --version apachejit-jit-v1
.venv/bin/python -m riskrank.deploy_module --seconds 60 --seed 42 --version flask-module-v3 --families lr rf lgbm
```

`flask-module-v3` is chosen from logistic regression, random forest and LightGBM, the set `flask-module-v2`
was chosen from. Both commands publish to `registry/<version>/`. To evaluate the published bundles instead of refitting them,
unpack the `v0.1.3` assets into `registry/` (and `models-flask-module-v2.tar.gz` from release `v0.1.2`
for `flask-module-v2`).

Flask module model, on its held-out snapshots (the pre-registered model is `flask-module-v2`):

```bash
.venv/bin/python -m riskrank.offline_module                          # -> runs/offline_flask-module-v2.json
.venv/bin/python -m riskrank.offline_module --model flask-module-v3  # -> runs/offline_flask-module-v3.json
```

Static-versus-process comparisons, the change-metric groups and the transfer check:

```bash
.venv/bin/python -m riskrank.static_vs_process flask --seconds 60 --seed 42
.venv/bin/python -m riskrank.static_vs_process groups --run runs/within-20261108T224535Z
.venv/bin/python -m riskrank.static_vs_process transfer
```

The `flask` and `transfer` modes read the commit-level corpus, so that corpus must be rebuilt first.

**First bounded check** (runtime-set defect recall on Flask):

```bash
.venv/bin/python -m riskrank.replay_first --seconds 60 --seed 42    # -> runs/first_check.json
```

**Second bounded check** (an executed, back-ported test). It needs `uv` on `PATH` to provide the
historical Pythons, and it ran on GitHub's Linux runners. Run the `second-check` workflow, then
`second-check-older` (Python 3.6 and 3.7 anchors, in containers), which merges both into one record.
Both read their inputs from release `eval-inputs-20261109`. Locally, the same phases are:

```bash
.venv/bin/python -m riskrank.replay_second screen
.venv/bin/python -m riskrank.replay_second rank
.venv/bin/python -m riskrank.replay_second combine --inputs <folder> <folder> ...
```

The result of record is `runs/second_check/combined-42622799420/second_check.json`.

**Benchmarks** (NASA MDP and PROMISE, protocol in `config/benchmarks.yaml`). The full run is too large
for one workstation and ran sharded on GitHub with the `benchmarks` workflow (run of record
`bench-actions-42609781856`). On one machine:

```bash
.venv/bin/python -m riskrank.run_benchmarks all --jobs 6
.venv/bin/python -m riskrank.run_benchmarks all --check --jobs 6    # small pipeline check, labelled as such
```

**Scale check** (5,000-item time targets, on a hosted runner): the `scale-check` workflow, or
`.venv/bin/python -m riskrank.scale_check --repeats 5 --out scale-check.json`.

### Flask replay (second deployment)

The fork `wisdomerhabor/flask` replays upstream Flask history after the fork commit `b78b5a21` in 34
batches. Each batch is one `workflow_dispatch` of the fork's `riskrank-replay.yml`. It merges the
batch's last upstream commit into `main` and pushes, which triggers the scoring workflow.

```bash
.venv/bin/python tools/replay_batches.py --log runs/replay-v3.jsonl
```

The driver uses `gh` and a local Flask clone at `data/repos/pallets/flask`. The fork name is set in
`tools/replay_batches.py` (`FORK`); change it to replay into your own fork. The fork needs a repository
secret `REPLAY_TOKEN`: a fine-grained token with write access to that fork's contents and workflow
files only. A push made with the workflow's own token would not trigger the scoring workflow.

To summarise the counted pushes, download each run's plan artifact (kept 7 days) and summarise:

```bash
gh run download <run-id> -R <owner>/flask -n riskrank-plan -D plans/<run-id>
.venv/bin/python -m riskrank.pipeline_summary plans --out flask-v3-replay-summary.json
```

To watch the results on the dashboard, see "Workstation side" above: `riskrank.ingest` with
`--repo <owner>/flask`.

### Results files

Attached to release `v1.0.0`, with a `SHA256SUMS` file:

- `flask-module-snapshots.zip`: the module-level Flask snapshot corpus (`flask-module-snapshots.csv`,
  1,213 rows: snapshot, module, tree commit, churn, changes, prior fixes, label, Radon version, size,
  complexity). It contains no author identifiers or developer metrics.
- `riskrank-v1.0.0-results.zip`: the aggregate results, at their `runs/` paths:
  - `offline_flask-module-v2.json`, `offline_flask-module-v3.json`: offline evaluation of the Flask models
  - `first_check.json`, `first_check_date.json`: first bounded check and its separation date
  - `second_check/combined-42622799420/second_check.json`: second bounded check
  - in both bounded-check files the upstream Flask commit identifiers are replaced with case numbers
    (`anchor-01`, `defect-01-fix` and so on), because they would link defects to developers; rebuild
    them with the commands above to recover the identifiers
  - `pipeline-checks/flask-v3-replay-summary.json`, `replay-v3.jsonl`: the 34-push Flask replay
    summary and its batch boundaries
  - `leakage-check.json`: the leakage check on the mined corpus
  - `deploy_module_flask-module-v3.json`, `deploy_jit_apachejit-jit-v1.json`: deployed-model selection
  - `within-20261108T224535Z/` (run of record) and the three sensitivity variants, and
    `cross-20261109T003200Z/`: ApacheJIT evaluations, manifests, family selection and Scott-Knott ESD
  - `bench-actions-42609781856/`: benchmark report, manifest, selection and Scott-Knott ESD results
  - `static_vs_process-*/`, `metric_groups-*/`, `transfer_check-*/`: exploratory comparisons

Model bundles are on release `v0.1.3`, and the second check's inputs (including the bug-fix list) are on
release `eval-inputs-20261109`.
