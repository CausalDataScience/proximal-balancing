"""Make `single_image` and the sealed project modules importable (code/ is not an installed package)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
