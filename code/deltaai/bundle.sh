#!/bin/bash
# Build the folder to rsync to DeltaAI.  Code, the frozen protocol if it exists, MNIST, nothing else.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-/tmp/scm7_v3_bundle}"
rm -rf "$out"; mkdir -p "$out/code" "$out/results/scm7_v3" "$out/results/scm7_v4" "$out/materials/benchmarks/mnist" "$out/logs"
cp "$here"/*.py "$out/code/"
cp -r "$here/deltaai" "$out/code/"
[ -f "$here/../results/scm7_v3/protocol.json" ] && cp "$here/../results/scm7_v3/protocol.json" "$out/results/scm7_v3/"
cp "$here/../results/scm7_v4/protocol.json" "$here/../results/scm7_v4/tolerance_calibration.json" "$out/results/scm7_v4/"
cp "$here/../materials/benchmarks/mnist/mnist.npz" "$out/materials/benchmarks/mnist/"
cp "$here/deltaai/scm7_v3.sbatch" "$here/deltaai/scm7_v3_largen.sbatch" "$here/deltaai/scm7_v4.sbatch" "$out/"
cat > "$out/requirements.txt" <<REQ
numpy
scipy
torch
REQ
cat > "$out/README.txt" <<TXT
On DeltaAI:
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  sed -i 's/REPLACE_WITH_ACCOUNT/<your allocation>/' scm7_v3.sbatch
  sbatch scm7_v3.sbatch
Afterwards, rsync results/scm7_v3/shards back; the local scorer reads them.
TXT
(cd "$out" && find . -type f ! -name MANIFEST.sha256 | sort | xargs shasum -a 256 > MANIFEST.sha256)
echo "bundle at $out ($(du -sh "$out" | cut -f1))"
