"""More replicates for the final experiments, run through the frozen runners
(memo/2026-09-23-replicate-extension-prd-v2.md).

No frozen file is edited.  Each frozen runner is imported unchanged; the driver only points its output
location at results/extension_v1/<experiment>/ and, for SCM-4, its device at the job's CUDA GPU.  Every task
runs in its own process with the thread count its sealed runs used, by explicit identifier from the
write-once protocol, which is frozen before any task runs.

    python3 -B run_extension_v1.py check-sources
    python3 -B run_extension_v1.py freeze
    python3 -B run_extension_v1.py run --cores 144 --gpus 2 --per-gpu 4 --time-limit-s 14400
    python3 -B run_extension_v1.py reproduce --root DIR [--items E6 E8 ...] --cores 2
    python3 -B run_extension_v1.py compare --root DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
EXT = RESULTS / "extension_v1"
PROTOCOL = EXT / "protocol_v1.json"
PRD = ROOT / "memo" / "2026-09-23-replicate-extension-prd-v2.md"
LABEL = "extension, run on DeltaAI after the protocol was frozen"

LEVELS = ("SCM-1", "SCM-2", "SCM-3")
E1E2_SEED = 99_700_000                     # + 100_000 i + 1_000 rep; i = 3 is SCM-4 (PRD section 2)
IDS = {"E1": range(20, 100), "E2": range(20, 100), "E3": range(21, 101), "E5": range(21, 111),
       "E6": range(22, 101), "E8": range(21, 101)}
THREADS = {"E1": 1, "E2": 2, "E3": 2, "E5": 2, "E6": 2, "E8": 2, "KSPC": 2}   # what the sealed runs used
GPU = {"E2"}                                # and the KSPC rows of E2, which rebuild the SCM-4 view
EXPECTED_MIN = {"SCM-3": 122, "SCM-2": 52, "SCM-1": 46, "E2": 20, "E5": 8, "E3": 4, "E8": 1.5, "E6": 1,
                "KSPC": 2}                   # Mac minutes x 2, used only to decide whether to launch near the limit
SEALED = {"E1": RESULTS / "family_v2" / "confirm", "E2": RESULTS / "family_v2" / "scm4" / "confirm",
          "E3": RESULTS / "rhc_bootstrap", "E5": RESULTS / "twins", "E6": RESULTS / "ihdp",
          "E8": RESULTS / "kspc_design", "KSPC": RESULTS / "kspc_comparator"}
REPRODUCE = [{"exp": "E6", "id": 12}, {"exp": "E8", "id": 1}, {"exp": "E3", "id": 1}, {"exp": "E5", "id": 11},
             {"exp": "E2", "level": "SCM-4", "rep": 6, "seed": 97_406_000},
             {"exp": "E1", "level": "SCM-1", "rep": 0, "seed": 97_200_000}]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------------------------- sources
def recorded_sources() -> dict[str, dict[str, str]]:
    """The code hashes each experiment's sealed results were produced with, from its own frozen record."""
    load = lambda p: json.loads(Path(p).read_text())
    e8 = dict(load(SEALED["E8"] / "replicate_1.json")["source_sha256"])
    e8.update({k: v for k, v in load(SEALED["E8"] / "FROZEN_v1_addendum.json")["current_sha256"].items()
               if k.endswith(".py") and k in e8})           # the addendum shows the current wrapper reproduces E8
    return {"E1": load(RESULTS / "family_v2" / "confirm_protocol_v1.json")["source_sha256"],
            "E2": load(RESULTS / "family_v2" / "scm4" / "confirm_protocol_v1.json")["source_sha256"],
            "E3": load(SEALED["E3"] / "boot_1.json")["source_sha256"],
            "E5": load(SEALED["E5"] / "replicate_11.json")["source_sha256"],
            "E6": load(SEALED["E6"] / "replicate_12.json")["source_sha256"],
            "E8": e8,
            "KSPC": load(SEALED["KSPC"] / "E1" / "SCM-1_replicate_0.json")["source_sha256"]}


# run_kspc_comparator.py was edited after its sealed rows were written (a docstring correction about shift
# invariance); the sealed version is not in git.  The reproduction of a sealed row, bit for bit, is the check.
KNOWN_EDITS = {"KSPC": {"run_kspc_comparator.py"}}


def check_sources() -> dict:
    out = {}
    for exp, rec in recorded_sources().items():
        bad = sorted(f for f, h in rec.items() if not (HERE / f).exists() or sha256(HERE / f) != h)
        out[exp] = {"files": len(rec), "mismatched": bad,
                    "unexplained": sorted(set(bad) - KNOWN_EDITS.get(exp, set()))}
    return out


# ----------------------------------------------------------------------------------------------- tasks
def e1e2_seed(i: int, rep: int) -> int:
    return E1E2_SEED + 100_000 * i + 1_000 * rep


def planned_tasks() -> list[dict]:
    """Every added task, in the order PRD section 5 fixes: SCM-3 first, SCM-4 next (it waits for GPU slots),
    then SCM-1 and SCM-2 interleaved, then the short experiments interleaved by identifier."""
    scm = {lv: [{"exp": "E1", "level": lv, "rep": r, "seed": e1e2_seed(i, r)} for r in IDS["E1"]]
           for i, lv in enumerate(LEVELS)}
    e2 = [{"exp": "E2", "level": "SCM-4", "rep": r, "seed": e1e2_seed(3, r)} for r in IDS["E2"]]
    out = scm["SCM-3"] + e2
    for a, b in zip(scm["SCM-1"], scm["SCM-2"]):
        out += [a, b]
    short = {e: [{"exp": e, "id": r} for r in IDS[e]] for e in ("E5", "E6", "E8", "E3")}
    for k in range(max(len(v) for v in short.values())):
        out += [v[k] for v in short.values() if k < len(v)]
    return out


def task_key(t: dict) -> str:
    if t["exp"] == "KSPC":
        return f"KSPC/{t['of']}/{t['name']}_replicate_{t['replicate']}"
    if t["exp"] in ("E1", "E2"):
        return f"{t['exp']}/{t['level']}_seed{t['seed']}"
    return f"{t['exp']}/{t['id']}"


def shard_of(t: dict, root: Path) -> Path:
    e = t["exp"]
    if e in ("E1", "E2"):
        return root / e / f"{t['level']}_seed{t['seed']}.json"
    if e == "E3":
        return root / e / f"boot_{t['id']}.json"
    if e == "KSPC":
        return root / "kspc" / t["of"] / f"{t['name']}_replicate_{t['replicate']}.json"
    return root / e / f"replicate_{t['id']}.json"


def kspc_child(t: dict, parent_shard: Path) -> dict | None:
    """The KSPC row of an E1, E2, E5 or E6 data set, pointed at that data set's own shard for the integrity
    check (COCA on W1 and W4 recomputed on the rebuilt data must equal the shard's values)."""
    if t["exp"] in ("E1", "E2"):
        return {"exp": "KSPC", "of": t["exp"], "name": t["level"], "replicate": t["rep"], "seed": t["seed"],
                "tau": 1.0, "sealed": str(parent_shard)}
    if t["exp"] in ("E5", "E6"):
        return {"exp": "KSPC", "of": t["exp"], "name": {"E5": "twins", "E6": "ihdp"}[t["exp"]],
                "replicate": t["id"], "seed": None, "tau": None, "sealed": str(parent_shard)}
    return None


# ----------------------------------------------------------------------------------------------- one task
def _use_cuda_if_present() -> str:
    import torch
    import family_v2_images as im
    if torch.cuda.is_available():
        im.device = lambda: torch.device("cuda")          # PRD section 4: the reader and pixel CNN on the GPU
    return str(im.device())


def run_one(t: dict, root: Path, proto_sha: str, skip_cevae_e1: bool = False) -> dict:
    e = t["exp"]
    if e == "E1":
        import run_family_v2 as rf
        rf.SHARDS["extension"] = root / "E1"
        if skip_cevae_e1:
            import family_v2_competitors as fc
            fc.cevae_ate = lambda view, seed: {"ate": None, "final_loss": None}
        thresholds = json.loads((RESULTS / "family_v2" / "confirm_protocol_v1.json").read_text())["screen_thresholds"]
        task = {"level": t["level"], "rep": t["rep"], "seed": t["seed"], "probe": True}
        return rf.run_task("extension", task, proto_sha, thresholds[t["level"]])
    if e == "E2":
        device = _use_cuda_if_present()
        import run_scm4 as r4
        r4.SHARDS["extension"] = root / "E2"
        threshold = json.loads((RESULTS / "family_v2" / "scm4" / "confirm_protocol_v1.json").read_text()
                               )["screen_thresholds"]["SCM-4"]
        task = {"level": t["level"], "rep": t["rep"], "seed": t["seed"]}
        return {**r4.run_task("extension", task, proto_sha, threshold), "device": device}
    if e == "E3":
        import rhc_bootstrap as rb
        rb.OUT = root / "E3"
        return rb.run_one(t["id"])
    if e == "E5":
        import run_twins as rtw
        rtw.RESULTS = root / "E5"
        return rtw.run_replicate(t["id"])
    if e == "E6":
        import run_ihdp_acic as ria
        ria.results_dir = lambda name: root / "E6"
        return ria.run_replicate("ihdp", t["id"])
    if e == "E8":
        import run_kspc_design as rkd
        rkd.RESULTS = root / "E8"
        return rkd.run_replicate(t["id"])
    if e == "KSPC":
        if t["of"] == "E2":
            _use_cuda_if_present()
        import run_kspc_comparator as rkc
        rkc.OUT = root / "kspc"
        task = {k: t[k] for k in ("name", "replicate", "seed", "tau", "sealed")} | {"exp": t["of"]}
        return rkc.run_task(task)
    raise ValueError(e)


def child_main(argv: list[str]) -> int:
    t, root, proto_sha, skip = json.loads(argv[0]), Path(argv[1]), argv[2], argv[3] == "1"
    t0 = time.time()
    status, info = "ok", {}
    try:
        info = run_one(t, root, proto_sha, skip)
    except BaseException as exc:                              # SystemExit from a runner's own check included
        status, info = "failed", {"error": f"{type(exc).__name__}: {exc}"}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    print("RESULT " + json.dumps({"key": task_key(t), "status": status, "seconds": round(time.time() - t0, 1),
                                  "max_rss_mb": round(rss_mb), "info": _short(info)}, default=str), flush=True)
    return 0 if status == "ok" else 1


def _short(info) -> dict:
    return {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, type(None)))}


# ----------------------------------------------------------------------------------------------- scheduler
def schedule(tasks: list[dict], root: Path, proto_sha: str, cores: int, gpus: int, per_gpu: int,
             time_limit_s: float | None, log_path: Path, skip_cevae_e1: bool = False, nice: int = 0) -> int:
    """First-fit on cores and GPU slots.  A finished E1, E2, E5 or E6 task queues its KSPC row at the front.
    A task is not launched when its expected length would pass the time limit."""
    start = time.time()
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d != ""]
    gpu_ids = visible[:gpus] if visible else [str(k) for k in range(gpus)]
    load = {g: 0 for g in gpu_ids}
    pending = [t for t in tasks if not shard_of(t, root).exists()]
    for t in tasks:                                      # a parent already done still owes its KSPC row
        child = kspc_child(t, shard_of(t, root))
        if child and shard_of(t, root).exists() and not shard_of(child, root).exists():
            pending.insert(0, child)
    running: dict = {}
    free = cores
    failures = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")

    def emit(rec: dict) -> None:
        log.write(json.dumps({"utc": now(), **rec}) + "\n")
        log.flush()

    emit({"event": "start", "tasks": len(pending), "cores": cores, "gpus": gpu_ids, "per_gpu": per_gpu,
          "time_limit_s": time_limit_s, "skip_cevae_e1": skip_cevae_e1})
    while pending or running:
        i = 0
        while i < len(pending) and free > 0:
            t = pending[i]
            exp = t["exp"]
            need = THREADS[exp]
            wants_gpu = exp in GPU or (exp == "KSPC" and t["of"] in GPU)
            gpu = None
            if wants_gpu and gpu_ids:
                gpu = min(load, key=load.get)
                if load[gpu] >= per_gpu:
                    i += 1
                    continue
            if need > free:
                i += 1
                continue
            est = 60 * EXPECTED_MIN.get(t.get("level") if exp == "E1" else exp, 5)
            if time_limit_s is not None and time.time() - start + est > time_limit_s - 120:
                i += 1
                continue
            pending.pop(i)
            env = dict(os.environ)
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env[var] = str(need)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
                load[gpu] += 1
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "_child", json.dumps(t), str(root), proto_sha,
                   "1" if skip_cevae_e1 else "0"]
            proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, preexec_fn=(lambda: os.nice(nice)) if nice else None)
            running[proc] = (t, need, gpu, time.time())
            free -= need
            emit({"event": "launch", "key": task_key(t), "gpu": gpu})
        if not running:
            if pending:
                emit({"event": "stop", "reason": "nothing more fits in the time limit",
                      "not_launched": [task_key(t) for t in pending]})
            break
        time.sleep(2)
        for proc in [p for p in running if p.poll() is not None]:
            t, need, gpu, t0 = running.pop(proc)
            out, err = proc.communicate()
            free += need
            if gpu is not None:
                load[gpu] -= 1
            result = next((json.loads(line[7:]) for line in out.splitlines() if line.startswith("RESULT ")),
                          {"key": task_key(t), "status": "failed", "info": {"error": "no result line"}})
            if result["status"] != "ok" or not shard_of(t, root).exists():
                failures += 1
                result["stderr_tail"] = err[-2000:]
            emit({"event": "done", **result, "wall_s": round(time.time() - t0, 1)})
            child = kspc_child(t, shard_of(t, root))
            if child and shard_of(t, root).exists() and not shard_of(child, root).exists():
                pending.insert(0, child)
    emit({"event": "end", "failures": failures, "elapsed_s": round(time.time() - start, 1)})
    log.close()
    return 1 if failures else 0


# ----------------------------------------------------------------------------------------------- protocol
def freeze() -> int:
    if PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} exists; write-once")
    checks = check_sources()
    if any(c["unexplained"] for c in checks.values()):
        raise SystemExit(f"source files differ from the frozen records: {checks}")
    tasks = planned_tasks()
    present = [task_key(t) for t in tasks if shard_of(t, EXT).exists()]
    if present:
        raise SystemExit(f"added shards already exist: {present[:5]}")
    import run_scm4_images as ri
    payload = {"artifact": "extension_v1_protocol", "frozen_utc": now(), "label": LABEL,
               "prd": {"path": str(PRD.relative_to(ROOT)), "sha256": sha256(PRD)},
               "tasks": tasks, "kspc_rows": "one per added E1, E2, E5 and E6 data set, queued when it finishes",
               "seed_rule_e1_e2": "99_700_000 + 100_000 i + 1_000 rep (i = 0, 1, 2 for SCM-1..3; 3 for SCM-4)",
               "threads": THREADS, "gpu_tasks": sorted(GPU) + ["KSPC rows of E2"],
               "device_policy": "SCM-4 reader and pixel CNN on CUDA when present (family_v2_images.device "
                                "assigned by this driver); everything else on CPU",
               "thresholds": {"E1": json.loads((RESULTS / "family_v2" / "confirm_protocol_v1.json").read_text()
                                               )["screen_thresholds"],
                              "E2": json.loads((RESULTS / "family_v2" / "scm4" / "confirm_protocol_v1.json"
                                                ).read_text())["screen_thresholds"]},
               "source_checks": checks, "known_edits": {k: sorted(v) for k, v in KNOWN_EDITS.items()},
               "reader_digests": ri.current_digests(),
               "driver_sha256": sha256(Path(__file__).resolve()),
               "current_sha256": {f.name: sha256(f) for f in sorted(HERE.glob("*.py"))
                                  if f.name in {n for c in recorded_sources().values() for n in c}},
               "budget_gpu_hours": {"smoke": 1.0, "main": 8.0, "reserve": 1.0, "total": 10.0}}
    EXT.mkdir(parents=True, exist_ok=True)
    with open(PROTOCOL, "x") as handle:
        handle.write(json.dumps(payload, indent=1) + "\n")
    print(f"frozen {PROTOCOL} with {len(tasks)} tasks; sha256 {sha256(PROTOCOL)}")
    return 0


# ----------------------------------------------------------------------------------------------- reproduce
VOLATILE = {"timing_s", "seconds", "generated_utc", "protocol_sha256", "stage", "source_sha256", "prereg_sha256",
            "device", "sealed", "task"}


def diff(a, b, path: str = "", out: list | None = None, volatile: set = VOLATILE) -> list:
    """Every leaf where the two shards differ, ignoring bookkeeping that records when and where a run was."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in volatile:
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", a.get(k, "<absent>"), b.get(k, "<absent>")))
            else:
                diff(a[k], b[k], f"{path}/{k}", out, volatile)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, volatile)
    elif a != b:
        out.append((path, a, b))
    return out


def reproduce_tasks(items: list[str] | None) -> list[dict]:
    rows = [r for r in REPRODUCE if items is None or r["exp"] in items]
    out = [dict(r) for r in rows]
    sealed_kspc = []
    for r in rows:                                      # the sealed KSPC rows of the reproduced data sets
        if r["exp"] == "E1":
            sealed_kspc.append({"exp": "KSPC", "of": "E1", "name": "SCM-1", "replicate": 0, "seed": 97_200_000,
                                "tau": 1.0, "sealed": str(SEALED["E1"] / "SCM-1_seed97200000.json")})
        if r["exp"] == "E6":
            sealed_kspc.append({"exp": "KSPC", "of": "E6", "name": "ihdp", "replicate": 12, "seed": None,
                                "tau": None, "sealed": str(SEALED["E6"] / "replicate_12.json")})
    return sealed_kspc + out


def sealed_path(t: dict) -> Path:
    e = t["exp"]
    if e in ("E1", "E2"):
        return SEALED[e] / f"{t['level']}_seed{t['seed']}.json"
    if e == "E3":
        return SEALED[e] / f"boot_{t['id']}.json"
    if e == "KSPC":
        return SEALED["KSPC"] / t["of"] / f"{t['name']}_replicate_{t['replicate']}.json"
    return SEALED[e] / f"replicate_{t['id']}.json"


def compare(root: Path, items: list[str] | None) -> dict:
    report = {}
    for t in reproduce_tasks(items):
        mine = shard_of(t, root)
        if not mine.exists():
            report[task_key(t)] = {"status": "missing"}
            continue
        d = diff(json.loads(sealed_path(t).read_text()), json.loads(mine.read_text()))
        report[task_key(t)] = {"status": "identical" if not d else "differs", "n_differences": len(d),
                               "first": [(p, str(x)[:80], str(y)[:80]) for p, x, y in d[:8]]}
    return report


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "_child":
        return child_main(argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("check-sources", "freeze", "run", "reproduce", "compare"))
    ap.add_argument("--root", type=Path, default=EXT)
    ap.add_argument("--items", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--time-limit-s", type=float)
    ap.add_argument("--skip-cevae-e1", action="store_true")
    ap.add_argument("--nice", type=int, default=0)
    a = ap.parse_args(argv[1:])
    if a.command == "check-sources":
        print(json.dumps(check_sources(), indent=1))
        return 0
    if a.command == "freeze":
        return freeze()
    if a.command == "compare":
        print(json.dumps(compare(a.root, a.items), indent=1, default=str))
        return 0
    if a.command == "reproduce":
        if a.root.resolve() == EXT.resolve():
            raise SystemExit("reproduce writes to a scratch --root, never to results/extension_v1")
        tasks = reproduce_tasks(a.items)
        return schedule(tasks, a.root, "reproduction-of-sealed-items", a.cores, a.gpus, a.per_gpu,
                        a.time_limit_s, a.root / "reproduce_log.jsonl", nice=a.nice)
    if not PROTOCOL.exists():
        raise SystemExit("freeze the protocol first")
    proto = json.loads(PROTOCOL.read_text())
    if proto["driver_sha256"] != sha256(Path(__file__).resolve()):
        raise SystemExit("the driver changed after the protocol was frozen")
    checks = check_sources()
    if any(c["unexplained"] for c in checks.values()):
        raise SystemExit(f"source files differ from the frozen records: {checks}")
    tasks = [t for t in proto["tasks"] if a.only is None or t["exp"] in a.only]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return schedule(tasks, EXT, sha256(PROTOCOL), a.cores, a.gpus, a.per_gpu, a.time_limit_s,
                    EXT / "logs" / f"run_{stamp}_{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl",
                    skip_cevae_e1=a.skip_cevae_e1, nice=a.nice)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
