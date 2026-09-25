"""Write one data set of the paper as an .npz file that PROBE.py reads.

Every file holds the learner's view, X, W1, ..., WJ, A and Y, and two numbers PROBE never reads as data:
`tau`, the true average treatment effect when it is known, and `probe_seed`, the seed PROBE used for this data
set in the paper (PROBE.py uses it when no --seed is given).

    python data/make_dataset.py scm   --level SCM-1 --rep 0 [--n 24000] --out scm1_rep0.npz
    python data/make_dataset.py scm4  --rep 0 [--n 24000] [--bank-dir raw/shapes3d] --out scm4_rep0.npz
    python data/make_dataset.py twins --replicate 11 [--raw-dir raw/twins] --out twins_11.npz
    python data/make_dataset.py ihdp  --replicate 12 [--raw-dir raw/ihdp] --out ihdp_12.npz
    python data/make_dataset.py acic  --replicate 2  [--raw-dir raw/acic2016] --out acic_2.npz
    python data/make_dataset.py rhc   [--csv raw/rhc/rhc.csv] --out rhc.npz
    python data/make_dataset.py kspc  --replicate 1  [--mnist raw/mnist/mnist.npz] --out kspc_1.npz

    python data/make_dataset.py scm4-prepare [--bank-dir raw/shapes3d]            (once, before scm4)
    python data/make_dataset.py scm4-train-reader [--bank-dir raw/shapes3d] --out reader.pt   (optional)

The seeds of the paper's first data sets are the defaults; --seed replaces the data seed of scm and scm4.
Relative paths are read from the current directory; the defaults are relative to the package root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # so that `python data/make_dataset.py` works from anywhere
    sys.path.insert(0, str(ROOT))

FOLD_SEED_OFFSET = 7                               # PROBE's fold seed is the data seed plus 7 (scm, scm4, twins, ihdp, acic)
PAPER_SCM_CONFIRM_SEED = 97_200_000                # SCM-1 to SCM-3: seed + 10,000 * level index + 1,000 * rep
PAPER_SCM4_CONFIRM_SEED = 97_400_000               # SCM-4: seed + 1,000 * rep


def save(out: str, view: dict, tau: float | None, probe_seed: int, **meta) -> None:
    arrays = {k: np.asarray(v) for k, v in view.items()}
    arrays["tau"] = np.array(np.nan if tau is None else float(tau))
    arrays["probe_seed"] = np.array(int(probe_seed))
    for key, value in meta.items():
        arrays[key] = np.array(value)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    blocks = sorted((k for k in view if k.startswith("W")), key=lambda k: int(k[1:]))
    shapes = ", ".join(f"{k} {view[k].shape[1]}" for k in blocks)
    print(f"wrote {out}: n = {len(view['A'])}, X {view['X'].shape[1]} columns, blocks {shapes}, "
          f"tau = {'unknown' if tau is None else f'{float(tau):.6f}'}, probe_seed = {probe_seed}")


def cmd_scm(args) -> None:
    from data import scm_family as g
    level = g.LEVELS[args.level]
    seed = args.seed if args.seed is not None else PAPER_SCM_CONFIRM_SEED + 10_000 * level.index + 1_000 * args.rep
    data = g.generate(level, args.n, seed)
    save(args.out, g.learner_view(data), g.TAU, seed + FOLD_SEED_OFFSET, setting=args.level, data_seed=seed)


def cmd_scm4(args) -> None:
    from data import scm4
    from data import scm_family as g
    seed = args.seed if args.seed is not None else PAPER_SCM4_CONFIRM_SEED + 1_000 * args.rep
    out = scm4.generate(args.n, seed, Path(args.bank_dir), Path(args.reader))
    print(f"reader errors by block: {json.dumps(out['reading'])}; device {out['device']}")
    save(args.out, out["view"], g.TAU, seed + FOLD_SEED_OFFSET, setting="SCM-4", data_seed=seed)


def cmd_scm4_prepare(args) -> None:
    from data import scm4
    manifest = scm4.prepare(Path(args.bank_dir))
    print(json.dumps({k: manifest[k] for k in ("h5", "cache_bytes", "split")}, indent=1))


def cmd_scm4_train_reader(args) -> None:
    from data import scm4
    record = scm4.train_reader(Path(args.bank_dir), Path(args.out))
    print(json.dumps({k: record[k] for k in ("fit", "excluded", "final_overall")}, indent=1))
    print("final gate:", "PASS" if record["final_gate"]["pass"] else "FAIL")


def cmd_twins(args) -> None:
    from data import twins as td
    if args.raw_dir:
        td.ROOT = Path(args.raw_dir)
    real = td.load_real()
    d = td.generate(real, args.replicate)
    view = {k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}
    save(args.out, view, d["tau"], td.REPLICATE_SEED + 1000 * args.replicate + FOLD_SEED_OFFSET,
         setting="Twins", replicate=args.replicate)


def cmd_ihdp_acic(args) -> None:
    from data import ihdp_acic as ia
    if args.raw_dir:
        if args.command == "ihdp":
            ia.IHDP_DIR = Path(args.raw_dir)
        else:
            ia.ACIC_DIR = Path(args.raw_dir)
    d = ia.DATASETS[args.command]["load"](args.replicate)
    view = {k: d[k] for k in ("X", "W1", "W2", "W3", "W4", "W5", "A", "Y")}
    save(args.out, view, d["tau"], ia.DATASETS[args.command]["seed"] + 1000 * args.replicate + FOLD_SEED_OFFSET,
         setting=args.command.upper(), replicate=args.replicate)


def cmd_rhc(args) -> None:
    from data import rhc as rd
    if args.csv:
        rd.CSV = Path(args.csv)
    data = rd.load()
    save(args.out, data["view"], None, rd.SEED, setting="RHC")


def cmd_kspc(args) -> None:
    from data import kspc_benchmark as kd
    if args.mnist:
        kd.MNIST = Path(args.mnist)
    d = kd.generate(args.replicate)
    view = {k: d[k] for k in ("X", "W1", "W2", "A", "Y")}
    save(args.out, view, d["tau"], kd.SEED + 1_000 * args.replicate, setting="KSPC", replicate=args.replicate)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write one data set of the paper as an .npz file for PROBE.py.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scm", help="SCM-1, SCM-2 or SCM-3")
    p.add_argument("--level", choices=("SCM-1", "SCM-2", "SCM-3"), default="SCM-1")
    p.add_argument("--n", type=int, default=24_000)
    p.add_argument("--rep", type=int, default=0, help="the paper's confirmation data set number, 0 to 19")
    p.add_argument("--seed", type=int, default=None, help="data seed; replaces the paper's seed of --rep")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scm)

    p = sub.add_parser("scm4", help="SCM-4, the image setting (needs torch and the prepared Shapes3D bank)")
    p.add_argument("--n", type=int, default=24_000)
    p.add_argument("--rep", type=int, default=0, help="the paper's confirmation data set number, 0 to 19")
    p.add_argument("--seed", type=int, default=None, help="data seed; replaces the paper's seed of --rep")
    p.add_argument("--bank-dir", default=str(ROOT / "raw" / "shapes3d"))
    p.add_argument("--reader", default=str(ROOT / "data" / "reader_frozen.pt"))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scm4)

    p = sub.add_parser("scm4-prepare", help="verify 3dshapes.h5, build the image cache and the bank split")
    p.add_argument("--bank-dir", default=str(ROOT / "raw" / "shapes3d"))
    p.set_defaults(func=cmd_scm4_prepare)

    p = sub.add_parser("scm4-train-reader", help="train a reader as in the paper (optional)")
    p.add_argument("--bank-dir", default=str(ROOT / "raw" / "shapes3d"))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scm4_train_reader)

    p = sub.add_parser("twins", help="Twins with the hidden-proxy recipe (replicates 11 to 20 in the paper)")
    p.add_argument("--replicate", type=int, default=11)
    p.add_argument("--raw-dir", default=None, help="folder with the three twin_pairs_*.csv files")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_twins)

    for name, first, what in (("ihdp", 12, "IHDP (replicates 12 to 21 in the paper)"),
                              ("acic", 2, "ACIC 2016 (realizations 2 to 10 in the paper)")):
        p = sub.add_parser(name, help=f"{what}, with the hidden-proxy recipe")
        p.add_argument("--replicate", type=int, default=first)
        p.add_argument("--raw-dir", default=None,
                       help="folder with the two ihdp_npci_1-100 npz files" if name == "ihdp"
                       else "folder with x.csv and zymu_<r>.csv")
        p.add_argument("--out", required=True)
        p.set_defaults(func=cmd_ihdp_acic)

    p = sub.add_parser("rhc", help="the observed right heart catheterization study")
    p.add_argument("--csv", default=None, help="path of rhc.csv")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_rhc)

    p = sub.add_parser("kspc", help="the kernel single-proxy benchmark with a binary treatment (replicates 1 to 20)")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--mnist", default=None, help="path of mnist.npz (Keras packaging)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_kspc)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
