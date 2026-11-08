#!/bin/sh
# Build the PySZZ image and run PySZZ's own R-SZZ oracle test (5 cases). Expect "+++ Test passed +++".
set -e
docker build -t pyszz:2026 "$(dirname "$0")"
docker run --rm --entrypoint sh -w /usr/src/app/test -e PYTHONPATH=/usr/src/app pyszz:2026 -c \
  'python3 -m zipfile -e repos_test.zip . && python3 test_rszz_main.py bugfix_commits_test.json ../conf/rszz.yml repos_test/ 2>&1 | grep -E "Test passed|Error"'
