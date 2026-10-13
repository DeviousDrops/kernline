#!/usr/bin/env bash
# tools/ceilings/run.sh: measure this GPU's ceilings and merge them into specs/<gpu>.json.
# Only the "device" and "ceilings" keys are replaced; hand-written keys (chip, driver, os, ...)
# are kept. Run on an otherwise idle, plugged-in laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ -x build/ceilings ] || { echo "build/ceilings missing; run cmake --build build" >&2; exit 1; }
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

json=$(build/ceilings)
"$PY" - "$json" <<'EOF'
import json, sys, datetime, pathlib
new = json.loads(sys.argv[1])
path = pathlib.Path("specs") / f"{new['gpu']}.json"
spec = json.loads(path.read_text()) if path.exists() else {"gpu": new["gpu"]}
spec["device"] = new["device"]
spec["ceilings"] = new["ceilings"] | {"measured": datetime.date.today().isoformat()}
path.write_text(json.dumps(spec, indent=2) + "\n")
print(path)
print(json.dumps(spec["ceilings"], indent=2))
EOF
