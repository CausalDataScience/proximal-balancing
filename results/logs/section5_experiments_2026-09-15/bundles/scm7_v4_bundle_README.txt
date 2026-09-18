On DeltaAI:
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  sed -i 's/REPLACE_WITH_ACCOUNT/<your allocation>/' scm7_v3.sbatch
  sbatch scm7_v3.sbatch
Afterwards, rsync results/scm7_v3/shards back; the local scorer reads them.
