"""Write the scoring service's models.json from the settings file (FR12): the versions to load and the
checksums they must have. Paths are as the service container sees them (/models/<version>).

    python -m riskrank.models_json riskrank.yml [container root, default /models] > models.json
"""
import json
import sys

import yaml


def main() -> None:
    s = yaml.safe_load(open(sys.argv[1]))
    root = sys.argv[2] if len(sys.argv) > 2 else "/models"
    out = {g: {"folder": f"{root}/{m['version']}", "sha256": m["sha256"]} for g, m in s["models"].items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
