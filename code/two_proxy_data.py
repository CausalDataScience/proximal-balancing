"""The two data sets whose proxy is a pair: the Zika outbreak in Brazil, and LaLonde's job training.

Both are real throughout.  Nothing is generated, nothing is hidden and reintroduced as a proxy: the
treatment, the covariates, the two proxy blocks and the outcome are all columns of the published file.
That is what makes them worth running and also what makes them hard, because neither supplies the exact
effect that the family of synthetic data sets supplies.

    Zika       673 Brazilian municipalities, 185 of them heavily exposed to the 2015-16 outbreak.
               W1 is the 2013 birth rate and W2 the 2014 birth rate, both before the outbreak; Y is the
               2016 rate; X is the three published municipality covariates.  There is no ground truth.
               Park, Richardson and Tchetgen Tchetgen report an effect on the treated of -2.180 (se
               0.342) using both proxies, and that is the number to sit beside, not a truth to score on.

    LaLonde    the 185 men treated in the National Supported Work demonstration, put against the 15,992
               CPS-SSA-1 controls.  W1 is 1974 earnings and W2 is 1975 earnings, the two pre-programme
               years that the literature treats as proxies for an unmeasured willingness to work; Y is
               1978 earnings; X is the six demographic covariates.  The randomised arm of the same
               experiment gives an effect on the treated of $1,794 (se 671), which is a real benchmark.

Neither data set says which of its two proxies may act on the treatment and which may act on the
outcome, and that is exactly the question PROBE is built not to need answered.

The propensity band used for the common-support reading is declared here, before any estimate is formed,
and the reason is written down beside it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
MATERIALS = HERE.parent / "materials" / "real_world_data"

ZIKA_CSV = MATERIALS / "zika" / "Zika_Brazil_2013.csv"
ZIKA_SHA = "a1a86446a6038ab79f7f29f767f54ae4e113288d9950c5ce75eeea59d7328664"
ZIKA_SEED = 98_700_000
ZIKA_PUBLISHED = {"estimand": "effect on the treated",
                  "semiparametric_coca_both_proxies": {"estimate": -2.180, "se": 0.342},
                  "crude": 3.38429585129043,
                  "source": "Park, Richardson and Tchetgen Tchetgen (2024), Biometrics 80(2) ujae027"}

LALONDE_TREATED = MATERIALS / "lalonde" / "nsw_dw.dta"
LALONDE_CONTROLS = MATERIALS / "lalonde" / "cps_controls.dta"
LALONDE_SEED = 98_800_000
LALONDE_PSID_CONTROLS = MATERIALS / "lalonde" / "psid_controls.dta"
LALONDE_PSID_SEED = 98_800_500
LALONDE_X = ["age", "education", "black", "hispanic", "married", "nodegree"]
LALONDE_BENCHMARK = {"estimand": "effect on the treated",
                     "estimate": 1794.34, "se": 670.99,
                     "source": "the randomised arm of the same experiment, as reported by Dehejia and "
                               "Wahba (1999) for the sample they use"}

# Declared before any estimate is formed.  PROBE's score is the one for the average effect, whose weights
# are 1/e and 1/(1-e), so a unit whose propensity is near zero or one contributes an unbounded term.  The
# band below bounds every weight by 1/0.05 = 20.  It changes the population: the reading on it is the
# average effect over the units whose propensity lies in the band, not over everyone.
BAND = (0.05, 0.95)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_zika() -> dict:
    got = sha256(ZIKA_CSV)
    if got != ZIKA_SHA:
        raise RuntimeError(f"{ZIKA_CSV} has sha256 {got}, expected {ZIKA_SHA}")
    d = pd.read_csv(ZIKA_CSV)
    wide = d.pivot(index="ID", columns="Time", values="Y").sort_index()
    meta = d[d.Time == 0].set_index("ID").sort_index()
    return {"X": meta[["X_1", "X_2", "X_3"]].to_numpy(float),
            "W1": wide[-1].to_numpy(float)[:, None],
            "W2": wide[0].to_numpy(float)[:, None],
            "Y": wide[1].to_numpy(float),
            "A": meta["Trt"].to_numpy(float),
            "names": {"W1": "2013 birth rate, before the outbreak",
                      "W2": "2014 birth rate, before the outbreak",
                      "Y": "2016 birth rate", "A": "heavy exposure to the 2015-16 outbreak"}}


def load_lalonde() -> dict:
    t = pd.read_stata(LALONDE_TREATED)
    t = t[t.treat == 1]
    c = pd.read_stata(LALONDE_CONTROLS)
    d = pd.concat([t, c], ignore_index=True)
    return {"X": d[LALONDE_X].to_numpy(float),
            "W1": d[["re74"]].to_numpy(float),
            "W2": d[["re75"]].to_numpy(float),
            "Y": d["re78"].to_numpy(float),
            "A": d["treat"].to_numpy(float),
            "names": {"W1": "1974 earnings, before the programme",
                      "W2": "1975 earnings, before the programme",
                      "Y": "1978 earnings", "A": "assigned to National Supported Work"}}


def load_lalonde_psid() -> dict:
    """The same 185 NSW men against the 2,490 PSID-1 controls, the second comparison group that LaLonde and
    Dehejia and Wahba report.  Same columns, same benchmark, same declared band."""
    t = pd.read_stata(LALONDE_TREATED)
    t = t[t.treat == 1]
    c = pd.read_stata(LALONDE_PSID_CONTROLS)
    d = pd.concat([t, c], ignore_index=True)
    return {"X": d[LALONDE_X].to_numpy(float),
            "W1": d[["re74"]].to_numpy(float),
            "W2": d[["re75"]].to_numpy(float),
            "Y": d["re78"].to_numpy(float),
            "A": d["treat"].to_numpy(float),
            "names": {"W1": "1974 earnings, before the programme",
                      "W2": "1975 earnings, before the programme",
                      "Y": "1978 earnings", "A": "assigned to National Supported Work"}}


LALONDE_DIR = MATERIALS / "lalonde"
AFDC_FILE = LALONDE_DIR / "NSW_AFDC_CS.dta"
AFDC_X = ["age", "educ", "nodegree", "married", "black", "hisp", "nchildren75"]
LALONDE_NAMES = {"W1": "1974 earnings, before the programme", "W2": "1975 earnings, before the programme",
                 "A": "assigned to National Supported Work"}


def load_lalonde_group(controls_file: str) -> dict:
    """The 185 NSW men of Dehejia and Wahba against one of LaLonde's comparison groups.  CPS-2 and CPS-3
    are subsets of CPS-1, and PSID-2 and PSID-3 of PSID-1, so these samples are nested, not independent."""
    t = pd.read_stata(LALONDE_TREATED)
    t = t[t.treat == 1]
    c = pd.read_stata(LALONDE_DIR / controls_file)
    d = pd.concat([t, c], ignore_index=True)
    return {"X": d[LALONDE_X].to_numpy(float), "W1": d[["re74"]].to_numpy(float),
            "W2": d[["re75"]].to_numpy(float), "Y": d["re78"].to_numpy(float), "A": d["treat"].to_numpy(float),
            "names": {**LALONDE_NAMES, "Y": "1978 earnings"}}


def _afdc_complete() -> "pd.DataFrame":
    """Calonico and Smith's reconstruction of the NSW AFDC women.  Fixed before any estimate: the outcome is
    1979 earnings, the convention of LaLonde (1986) and Calonico and Smith (2017) for this sample; only
    complete cases on 1974, 1975 and 1979 earnings are used, for the benchmark and the observational
    samples alike.  `afdc75` is not a covariate: every experimental woman received AFDC, so conditioning on
    it would leave no overlap with the PSID women."""
    a = pd.read_stata(AFDC_FILE)
    return a.dropna(subset=["re74", "re75", "re79"])


def afdc_benchmark() -> dict:
    a = _afdc_complete()
    exp = a[a.sample_dw == 1]
    t, c = exp[exp.treated == 1].re79, exp[exp.treated == 0].re79
    return {"estimand": "effect on the treated", "estimate": float(t.mean() - c.mean()),
            "se": float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))),
            "n_treated": int(len(t)), "n_control": int(len(c)),
            "source": "the randomised arm of the NSW AFDC women in Calonico and Smith's reconstruction, complete "
                      "cases on 1974, 1975 and 1979 earnings"}


def afdc_early_benchmark() -> dict:
    """The early random-assignment women (Smith and Todd's restriction, as Calonico and Smith use it): the
    same complete-case rule, the randomised contrast within that group."""
    a = _afdc_complete()
    exp = a[a.sample_early == 1]
    t, c = exp[exp.treated == 1].re79, exp[exp.treated == 0].re79
    return {"estimand": "effect on the treated", "estimate": float(t.mean() - c.mean()),
            "se": float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))),
            "n_treated": int(len(t)), "n_control": int(len(c)),
            "source": "the randomised arm of the early random-assignment NSW AFDC women, complete cases on 1974, "
                      "1975 and 1979 earnings"}


def load_afdc(comparison: str, treated_sample: str = "sample_dw") -> dict:
    """The experimental AFDC treated women against PSID-1 or PSID-2 women.  PSID-2 is a subset of PSID-1."""
    a = _afdc_complete()
    t = a[(a[treated_sample] == 1) & (a.treated == 1)].assign(A=1.0)
    c = a[a[comparison] == 1].assign(A=0.0)
    d = pd.concat([t, c], ignore_index=True)
    return {"X": d[AFDC_X].to_numpy(float), "W1": d[["re74"]].to_numpy(float),
            "W2": d[["re75"]].to_numpy(float), "Y": d["re79"].to_numpy(float), "A": d["A"].to_numpy(float),
            "names": {**LALONDE_NAMES, "Y": "1979 earnings"}}


def propensity(view: dict) -> np.ndarray:
    z = np.column_stack([view["X"], view["W1"], view["W2"]])
    z = (z - z.mean(0)) / (z.std(0) + 1e-12)
    return LogisticRegression(max_iter=5000).fit(z, view["A"]).predict_proba(z)[:, 1]


def restrict(view: dict, keep: np.ndarray) -> dict:
    out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in view.items()}
    out["names"] = view["names"]
    return out


def drop_dependent_columns(view: dict, labels: list[str]) -> tuple[dict, list[str]]:
    """Remove covariate columns that the intercept and the other columns already span.

    Restricting to a band can make two indicators exactly complementary: on LaLonde's band every kept man
    is either black or hispanic, so those two columns and the intercept are linearly dependent and any
    design built from them is singular.  That is a property of the restricted sample, not of one
    estimator, so the columns are dropped once, for every estimator alike, and the names are recorded.
    A column-pivoted QR keeps a maximal independent subset in the original order."""
    x = view["X"]
    design = np.column_stack([np.ones(len(x)), (x - x.mean(0)) / (x.std(0) + 1e-12)])
    rank = int(np.linalg.matrix_rank(design))
    if rank == design.shape[1]:
        return view, []
    keep, dropped = [], []
    current = np.ones((len(x), 1))
    for j in range(x.shape[1]):
        trial = np.column_stack([current, (x[:, j] - x[:, j].mean()) / (x[:, j].std() + 1e-12)])
        if np.linalg.matrix_rank(trial) > np.linalg.matrix_rank(current):
            current, _ = trial, keep.append(j)
        else:
            dropped.append(labels[j])
    out = dict(view)
    out["X"] = x[:, keep]
    return out, dropped


DATASETS = {"zika": {"load": load_zika, "seed": ZIKA_SEED, "reference": ZIKA_PUBLISHED,
                     "covariate_labels": ["X_1", "X_2", "X_3"]},
            "lalonde": {"load": load_lalonde, "seed": LALONDE_SEED, "reference": LALONDE_BENCHMARK,
                        "covariate_labels": LALONDE_X},
            "lalonde_psid": {"load": load_lalonde_psid, "seed": LALONDE_PSID_SEED, "reference": LALONDE_BENCHMARK,
                             "covariate_labels": LALONDE_X}}
# Nested comparison groups and the AFDC women, added 2026-09-22 20:30 CDT with seeds fixed before running.
for _name, _file, _seed in (("lalonde_cps2", "cps_controls2.dta", 98_800_200),
                            ("lalonde_cps3", "cps_controls3.dta", 98_800_300),
                            ("lalonde_psid2", "psid_controls2.dta", 98_800_600),
                            ("lalonde_psid3", "psid_controls3.dta", 98_800_700)):
    DATASETS[_name] = {"load": (lambda f=_file: load_lalonde_group(f)), "seed": _seed,
                       "reference": LALONDE_BENCHMARK, "covariate_labels": LALONDE_X}
for _name, _flag, _seed in (("afdc_psid1", "psid1", 98_800_800), ("afdc_psid2", "psid2", 98_800_900)):
    DATASETS[_name] = {"load": (lambda f=_flag: load_afdc(f)), "seed": _seed,
                       "reference": afdc_benchmark(), "covariate_labels": AFDC_X}
# The early random-assignment women against the same two comparison groups, added 2026-09-22 21:30 CDT
# with seeds fixed before running.  They lie inside the women above, so they are nested too.
for _name, _flag, _seed in (("afdc_early_psid1", "psid1", 98_801_000), ("afdc_early_psid2", "psid2", 98_801_100)):
    DATASETS[_name] = {"load": (lambda f=_flag: load_afdc(f, "sample_early")), "seed": _seed,
                       "reference": afdc_early_benchmark(), "covariate_labels": AFDC_X}
