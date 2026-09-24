#!/bin/bash
# The files the replicate extension needs on DeltaAI (memo/2026-09-23-replicate-extension-prd-v2.md, section 4),
# as paths relative to the project root, and their sha256 manifest.  Nothing is copied here; the list feeds
#   rsync -a --files-from=<out>/files.txt <project root>/ deltaai:/work/hdd/bgtp/yjung6/probe_ext_v1/
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?usage: extension_v1_files.sh <output folder>}"
mkdir -p "$out"
cd "$root"
{
  ls code/*.py
  ls code/deltaai/extension_v1_*.sbatch
  find materials/external_code/KernelSingleProxy -type f ! -path "*/__pycache__/*" ! -name "*.pyc" ! -name ".DS_Store"
  for f in 3dshapes_images_uint8.npy reader_frozen.pt reader_frozen.json reader_final_check.json split_rows.npz manifest.json; do
    echo "materials/benchmarks/shapes3d/$f"; done
  echo materials/benchmarks/mnist/mnist.npz
  for f in X T Y; do echo "materials/real_world_data/twins/twin_pairs_${f}_3years_samesex.csv"; done
  echo materials/real_world_data/ihdp/ihdp_npci_1-100.train.npz
  echo materials/real_world_data/ihdp/ihdp_npci_1-100.test.npz
  echo materials/real_world_data/rhc/rhc.csv
  echo materials/real_world_data/lalonde/NSW_AFDC_CS.dta     # read when two_proxy_data is imported (E3, E8)
  for f in 2026-09-18-family-v2-role-blind-dimension-prd-v1.md 2026-09-20-scm4-shapes3d-reader-spec-v1.md \
           2026-09-22-twins-real-outcome-prereg-v1.md 2026-09-22-ihdp-acic-prereg-v1.md \
           2026-09-22-rhc-real-data-prereg-v1.md 2026-09-23-replicate-extension-prd-v2.md; do echo "memo/$f"; done
  for f in family_v2/confirm_protocol_v1.json family_v2/dev_protocol_v1.json \
           family_v2/scm4/confirm_protocol_v1.json family_v2/scm4/dev_protocol_v1.json \
           family_v2/confirm/SCM-1_seed97200000.json family_v2/scm4/confirm/SCM-4_seed97406000.json \
           rhc_bootstrap/boot_1.json twins/replicate_11.json ihdp/replicate_12.json \
           kspc_design/replicate_1.json kspc_design/FROZEN_v1_addendum.json \
           kspc_comparator/E1/SCM-1_replicate_0.json kspc_comparator/E6/ihdp_replicate_12.json \
           extension_v1/protocol_v1.json; do echo "results/$f"; done
} > "$out/files.txt"
missing=$(while read -r f; do [ -f "$f" ] || echo "$f"; done < "$out/files.txt")
[ -z "$missing" ] || { echo "missing: $missing"; exit 1; }
xargs shasum -a 256 < "$out/files.txt" > "$out/MANIFEST.sha256"
echo "$(wc -l < "$out/files.txt") files, $(xargs du -ch < "$out/files.txt" | tail -1 | cut -f1) -> $out"
