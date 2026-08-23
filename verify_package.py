#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import json

root = Path(__file__).resolve().parent
module_path = root / "code" / "final_experiments" / "verify_manifest.py"
spec = importlib.util.spec_from_file_location("spb_verify_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(json.dumps(module.verify(root, root / "MANIFEST.json"), indent=2))
