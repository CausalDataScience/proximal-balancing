"""E11-v2: a literal audit of Algorithm 1 on an exact finite-state positive control.

Why a new version.  E11-v1 decided theta_S = tau by a numerical root tolerance and read the radius off a
200,000-row Monte Carlo draw.  Neither is exact, so v1 is kept as a numerical positive-control audit and is
not used for a claim about the theorem.  Here every quantity the theorem names is computed in exact
arithmetic over a finite state space, and equality of candidate values is decided by the map identity rather
than by the distance between two floats.

The control.  U and I are independent fair coins, X is identically zero, and

    P(A = 1 | U) = 1/4 + U/2,      Y(a) = a - 2U,      so tau = 1.

The proxy blocks are W_1 = W_2 = W_3 = U and W_4 = I, so J = 4 and T = 2^4 - 2 = 14.  Every observable is a
function of (U, I, A), which takes eight values, so expectations are finite sums of rationals.

The encoder class of each split is a singleton, which is the smallest case Algorithm 1 admits.  If both the
held-out set and its complement contain a copy of U, the representation is the lowest-indexed copy of U in
the complement; otherwise it is the constant zero.  That yields three kinds of candidate:

    twelve splits whose representation is U          theta_S = 1, population Brier gap 0
    the split {4}                                    theta_S = 0, population Brier gap 0
    the split {1,2,3}                                theta_S = 0, population Brier gap 1/16

The split {4} is the point of the design: it balances exactly and still carries the wrong value, so the
screen cannot remove it and the plurality step has to.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import product

import numpy as np

import minimal_suite_common as c

N_BLOCKS_V2 = 4
T_V2 = 2 ** N_BLOCKS_V2 - 2
TAU = F(1)
MAP_U = "U"
MAP_BAD = "CONST_BAD_BALANCED"
MAP_UNBAL = "CONST_UNBALANCED"


def splits_v2() -> list[tuple[int, ...]]:
    """Blocks are numbered 1..4 here, matching the manuscript's W_1..W_4."""
    out = []
    for mask in range(1, 2 ** N_BLOCKS_V2 - 1):
        out.append(tuple(j + 1 for j in range(N_BLOCKS_V2) if mask >> j & 1))
    return sorted(out, key=lambda s: (len(s), s))


U_BLOCKS = (1, 2, 3)


def encoder_map(split: tuple[int, ...]) -> str:
    """The singleton encoder of each split, and with it the exact identity of its candidate value.

    The rule uses only which blocks are held out, never a validity label and never tau.
    """
    held, kept = set(split), set(range(1, N_BLOCKS_V2 + 1)) - set(split)
    u_kept = sorted(b for b in kept if b in U_BLOCKS)
    u_held = sorted(b for b in held if b in U_BLOCKS)
    if u_kept and u_held:
        return MAP_U
    return MAP_BAD if split == (4,) else MAP_UNBAL


def representation(split: tuple[int, ...], u: int) -> int:
    """Z_S realised at a state.  Either the retained copy of U, or the constant zero."""
    return u if encoder_map(split) == MAP_U else 0


def heldout_values(split: tuple[int, ...], u: int, i: int) -> tuple[int, ...]:
    """T_S = (X, W_S) with X identically zero, so only the held-out blocks carry information."""
    return tuple(u if b in U_BLOCKS else i for b in split)


# ----------------------------------------------------------------------------- the exact state space
def states() -> list[dict]:
    """The eight equally reachable states of (U, I, A) with their exact probabilities."""
    out = []
    for u, i in product((0, 1), repeat=2):
        p_treat = F(1, 4) + F(u, 2)
        for a in (0, 1):
            out.append({"U": u, "I": i, "A": a,
                        "p": F(1, 4) * (p_treat if a else 1 - p_treat),
                        "Y": F(a) - 2 * u})
    assert sum(s["p"] for s in out) == 1
    return out


STATES = states()


def exact_theta(split: tuple[int, ...]) -> F:
    """theta_S = E[ E(Y | A=1, Z_S) - E(Y | A=0, Z_S) ], as an exact rational."""
    cells: dict[int, dict[int, list[F]]] = {}
    for s in STATES:
        z = representation(split, s["U"])
        cells.setdefault(z, {0: [F(0), F(0)], 1: [F(0), F(0)]})
        cells[z][s["A"]][0] += s["p"] * s["Y"]
        cells[z][s["A"]][1] += s["p"]
    total = F(0)
    for z, arms in cells.items():
        mass = sum(arms[a][1] for a in (0, 1))
        m1 = arms[1][0] / arms[1][1]
        m0 = arms[0][0] / arms[0][1]
        total += mass * (m1 - m0)
    return total


def exact_brier_gap(split: tuple[int, ...]) -> F:
    """The population nested Brier risk gap of the split's encoder, exactly.

    The base critic is the true P(A=1 | Z_S) and the augmented critic the true P(A=1 | T_S, Z_S), both of
    which are exact conditional means on a finite state space, so the gap is the mean squared difference of
    two rationals.
    """
    def conditional(keyfn) -> dict:
        num, den = {}, {}
        for s in STATES:
            k = keyfn(s)
            num[k] = num.get(k, F(0)) + s["p"] * s["A"]
            den[k] = den.get(k, F(0)) + s["p"]
        return {k: num[k] / den[k] for k in num}

    base = conditional(lambda s: representation(split, s["U"]))
    aug = conditional(lambda s: (representation(split, s["U"]), heldout_values(split, s["U"], s["I"])))
    gap = F(0)
    for s in STATES:
        z = representation(split, s["U"])
        p0 = base[z]
        p1 = aug[(z, heldout_values(split, s["U"], s["I"]))]
        gap += s["p"] * ((p0 - p1) ** 2)
    return gap


def design_table() -> list[dict]:
    rows = []
    for sp in splits_v2():
        rows.append({"split": list(sp), "split_id": c.split_id(sp), "map_id": encoder_map(sp),
                     "theta_S": exact_theta(sp), "population_gap": exact_brier_gap(sp)})
    return rows


# ----------------------------------------------------------------------------- exact nuisance quantities
def fit_saturated(rows: dict, split: tuple[int, ...], eta: F) -> dict:
    """Saturated finite-cell propensity and arm means from the N sample, with the propensity clipped.

    The representation takes two values at most, so the model is a cell mean per (z, arm).  An empty cell
    falls back to the marginal, which keeps the fit defined without inventing information.
    """
    z = np.array([representation(split, u) for u in rows["U"]])
    a, y = rows["A"], rows["Y"]
    e_hat, m_hat = {}, {}
    for zv in (0, 1):
        sel = z == zv
        if sel.sum() == 0:
            e_hat[zv] = F(int(a.sum()), max(len(a), 1))
        else:
            e_hat[zv] = F(int(a[sel].sum()), int(sel.sum()))
        e_hat[zv] = min(max(e_hat[zv], eta), 1 - eta)
        for arm in (0, 1):
            cell = sel & (a == arm)
            if cell.sum() == 0:
                pool = a == arm
                m_hat[(zv, arm)] = F(float(y[pool].mean())).limit_denominator(10 ** 9) if pool.sum() else F(0)
            else:
                m_hat[(zv, arm)] = F(int(round(float(y[cell].sum()) * 2)), 2 * int(cell.sum()))
    return {"e": e_hat, "m": m_hat}


def exact_true_nuisance(split: tuple[int, ...]) -> dict:
    """The true e_S and m_{a,S} on the finite state space, as exact rationals."""
    num_a, den, num_y = {}, {}, {}
    for s in STATES:
        z = representation(split, s["U"])
        den[z] = den.get(z, F(0)) + s["p"]
        num_a[z] = num_a.get(z, F(0)) + s["p"] * s["A"]
        key = (z, s["A"])
        num_y[key] = num_y.get(key, [F(0), F(0)])
        num_y[key][0] += s["p"] * s["Y"]
        num_y[key][1] += s["p"]
    e = {z: num_a[z] / den[z] for z in den}
    m = {k: v[0] / v[1] for k, v in num_y.items()}
    return {"e": e, "m": m}


def exact_radius_terms(split: tuple[int, ...], fitted: dict, eta: F) -> dict:
    """q_e, q_0, q_1 and sigma for one split, by enumerating the eight states with exact arithmetic.

    The norms are taken under the distribution of the representation, and the score variance is the
    conditional variance of the plug-in influence function given the two learning samples.  No Monte Carlo
    evaluator appears anywhere.
    """
    truth = exact_true_nuisance(split)
    z_mass: dict[int, F] = {}
    for s in STATES:
        z = representation(split, s["U"])
        z_mass[z] = z_mass.get(z, F(0)) + s["p"]
    q_e2 = sum(mass * (fitted["e"][z] - truth["e"][z]) ** 2 for z, mass in z_mass.items())
    q_a2 = {}
    for arm in (0, 1):
        q_a2[arm] = sum(mass * (fitted["m"][(z, arm)] - truth["m"][(z, arm)]) ** 2
                        for z, mass in z_mass.items())
    mean, second = F(0), F(0)
    for s in STATES:
        z = representation(split, s["U"])
        e = fitted["e"][z]
        m1, m0 = fitted["m"][(z, 1)], fitted["m"][(z, 0)]
        uif = m1 - m0 + (F(s["A"]) * (s["Y"] - m1)) / e - (F(1 - s["A"]) * (s["Y"] - m0)) / (1 - e)
        mean += s["p"] * uif
        second += s["p"] * uif ** 2
    var = second - mean ** 2
    return {"q_e2": q_e2, "q_02": q_a2[0], "q_12": q_a2[1], "sigma2": var, "score_mean": mean}


def _sqrt_up(x: F, digits: int = 60) -> F:
    """Square root rounded upward, with the result checked to be an upper bound before it is returned.

    The scaled value must be rounded up, not down, before the integer square root is taken.  Taking the
    floor first lets the case where that floor is exactly a perfect square slip through: for
    x = 1 + 10^-121 the floor is (10^60)^2, the routine returns exactly 1, and 1^2 < x.  The certificate
    this radius feeds is one sided, so a single such input would invalidate it.  The final assertion makes
    the property a post-condition rather than an argument about the arithmetic.
    """
    if x <= 0:
        return F(0)
    scale = F(10) ** (2 * digits)
    n = -((-(x * scale)).numerator // (-(x * scale)).denominator)  # ceiling of x * scale
    r = int(np.floor(np.sqrt(float(n)))) if n < 2 ** 52 else None
    if r is None:
        lo, hi = 0, 1 << ((n.bit_length() + 1) // 2 + 1)
        while lo < hi:
            mid = (lo + hi) // 2
            if mid * mid < n:
                lo = mid + 1
            else:
                hi = mid
        r = lo
    while r * r < n:
        r += 1
    out = F(r, 10 ** digits)
    if out * out < x:
        raise AssertionError(f"_sqrt_up returned a value below the true root for {x}")
    return out


def exact_rho(terms: dict[tuple, dict], n_e: int, delta: F, n_screened: int, eta: F) -> dict:
    """rho = max over the screened splits of sigma / sqrt(n_E delta / N) + q_e (q_0 + q_1) / eta.

    The variance term is taken as one rounded root of the exact ratio sigma^2 N / (n_E delta), not as a
    quotient of two separately rounded roots.  Rounding the denominator upward shrinks the quotient, so
    the two-root form is not one sided: with sigma^2 = 1 and the denominator sqrt(2) it returns a value
    below the truth.  The bias term is a product of upward-rounded roots of non-negative quantities, which
    is an upper bound once each root is one, and eta is exact.  Both terms are checked below.
    """
    scale = F(n_screened) / (F(n_e) * delta)
    per = {}
    for sp, t in terms.items():
        variance_term = _sqrt_up(t["sigma2"] * scale)
        if variance_term * variance_term < t["sigma2"] * scale:
            raise AssertionError(f"variance term is not an upper bound for {sp}")
        q_e, q_0, q_1 = _sqrt_up(t["q_e2"]), _sqrt_up(t["q_02"]), _sqrt_up(t["q_12"])
        bias_term = q_e * (q_0 + q_1) / eta
        for root, square in ((q_e, t["q_e2"]), (q_0, t["q_02"]), (q_1, t["q_12"])):
            if root * root < square:
                raise AssertionError(f"a nuisance root is not an upper bound for {sp}")
        per[sp] = {"variance_term": variance_term, "bias_term": bias_term,
                   "epsilon": variance_term + bias_term}
    worst = max(per, key=lambda s: per[s]["epsilon"])
    return {"rho": per[worst]["epsilon"], "argmax": worst, "per_split": per,
            "rounding": "one upward-rounded root per term, each verified to be an upper bound"}


# ----------------------------------------------------------------------------- Algorithm 1, literally
def screen_split(rows: dict, split: tuple[int, ...], fitted: dict, t: float, eta: F) -> dict:
    """Line 3: keep the split if its empirical Brier gap is at most t and the overlap holds.

    The gap is the saturated-cell estimate of max(R0 - R1, 0) on the screening rows, which for this control
    is again a finite-cell computation.
    """
    z = np.array([representation(split, u) for u in rows["U"]])
    tv = [tuple(heldout_values(split, u, i)) for u, i in zip(rows["U"], rows["I"])]
    a = rows["A"]

    def cell_mean(keys):
        num, den = {}, {}
        for k, av in zip(keys, a):
            num[k] = num.get(k, 0.0) + av
            den[k] = den.get(k, 0) + 1
        return np.array([num[k] / den[k] for k in keys])

    p0 = cell_mean([int(v) for v in z])
    p1 = cell_mean([(int(zz), tt) for zz, tt in zip(z, tv)])
    r0 = float(np.mean((a - p0) ** 2))
    r1 = float(np.mean((a - p1) ** 2))
    gap = max(r0 - r1, 0.0)
    overlap = all(float(eta) <= float(fitted["e"][zv]) <= 1.0 - float(eta) for zv in fitted["e"])
    return {"gap": gap, "R0": r0, "R1": r1, "overlap_ok": overlap,
            "retained": bool(gap <= t and overlap)}


def aipw_estimate(rows: dict, split: tuple[int, ...], fitted: dict) -> float:
    """Line 7: the honest AIPW estimate on the evaluation rows with the frozen nuisances."""
    z = np.array([representation(split, u) for u in rows["U"]])
    a, y = rows["A"], rows["Y"]
    e = np.array([float(fitted["e"][int(zz)]) for zz in z])
    m1 = np.array([float(fitted["m"][(int(zz), 1)]) for zz in z])
    m0 = np.array([float(fitted["m"][(int(zz), 0)]) for zz in z])
    return float(np.mean(m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)))


def link_and_return(labels: list, estimates: np.ndarray, rho: float) -> dict:
    """Lines 8 and 9: link within 2 rho, take the median of the largest component, ties give a set."""
    if len(estimates) == 0:
        return {"return_kind": "no_return", "output": None, "candidates": [], "members": []}
    parent = list(range(len(estimates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            if abs(estimates[i] - estimates[j]) <= 2.0 * rho:
                parent[find(i)] = find(j)
    comp: dict[int, list[int]] = {}
    for i in range(len(estimates)):
        comp.setdefault(find(i), []).append(i)
    biggest = max(map(len, comp.values()))
    largest = [v for v in comp.values() if len(v) == biggest]
    cands = [float(np.median(estimates[idx])) for idx in largest]
    return {"return_kind": "unique_largest" if len(largest) == 1 else "tied_largest",
            "output": cands[0] if len(cands) == 1 else None, "candidates": cands,
            "members": [[labels[i] for i in idx] for idx in largest],
            "component_sizes": sorted((len(v) for v in comp.values()), reverse=True)}


def sampling_terms(T: int, N: int, mm: int, gap: F) -> dict:
    """P(R=0) and E[exp(-R Delta^2 / 2)] as exact finite sums over the hypergeometric law."""
    from math import comb, exp
    total = comb(T, mm)
    p_zero = comb(T - N, mm) / total if T - N >= mm else 0.0
    g2 = float(gap) ** 2 / 2.0
    expectation = sum(comb(N, r) * comb(T - N, mm - r) * exp(-r * g2)
                      for r in range(0, min(mm, N) + 1) if 0 <= mm - r <= T - N) / total
    return {"P_R_zero": p_zero, "E_exp": expectation}


def theorem_bound(T: int, N: int, mm: int, delta: F, gap: F, L: int) -> dict:
    s = sampling_terms(T, N, mm, gap)
    return {"B": 1.0 - float(delta) - s["P_R_zero"] - L * s["E_exp"], **s}
