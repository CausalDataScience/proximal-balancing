import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probe_structured_scm import (
    ALL_SPLITS,
    ROOT_TIE_TOL,
    SCMSpec,
    Transform,
    aggregate,
    array_sha256,
    brier_audit,
    brier_select,
    generate_role,
    observed_view,
    moment_root,
    partial_covariance,
    representation,
    split_category,
    write_json_atomic,
)


class StructuredSCMTests(unittest.TestCase):
    def test_all_30_and_original_topology_counts(self):
        self.assertEqual(len(ALL_SPLITS), 30)
        counts = {key: 0 for key in ("clean", "bad_singleton", "mixed", "S0")}
        for split in ALL_SPLITS:
            counts[split_category(split)] += 1
        self.assertEqual(counts, {"clean": 7, "bad_singleton": 1, "mixed": 7, "S0": 15})

    def test_total_role_budget_is_literal(self):
        spec = SCMSpec("test", 1.5, 0.9, 2.7, -0.6)
        roles = [generate_role(spec, 1500, 10 + j) for j in range(4)]
        self.assertEqual(sum(len(role["A"]) for role in roles), 6000)
        self.assertEqual(len({10 + j for j in range(4)}), 4)

    def test_rotation_uses_each_fold_once_per_role(self):
        rotations = [[(r + j) % 4 for j in range(4)] for r in range(4)]
        for row in rotations:
            self.assertEqual(sorted(row), [0, 1, 2, 3])
        for role in range(4):
            self.assertEqual(sorted(row[role] for row in rotations), [0, 1, 2, 3])

    def test_learner_view_strips_truth(self):
        spec = SCMSpec("test", 1.5, 0.9, 2.7, -0.6)
        raw = generate_role(spec, 100, 1)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        self.assertNotIn("U_eval_only", view)
        self.assertNotIn("p_eval_only", view)
        self.assertNotIn("Y", view)

    def test_multivariate_moment_fit_is_heldout_order_invariant(self):
        spec = SCMSpec("test", 1.5, 0.9, 2.7, -0.6)
        raw = generate_role(spec, 4000, 17)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        forward = moment_root(view, (1, 2))
        reverse = moment_root(view, (2, 1))
        self.assertAlmostEqual(forward["r"], reverse["r"], places=8)
        self.assertAlmostEqual(forward["objective"], reverse["objective"], places=12)

    def test_global_i_as_x_topology_has_no_joint_balance_root(self):
        """Regression test for the rejected 15/1/14 symmetric proposal."""
        rng = np.random.default_rng(4)
        n = 200_000
        u, inst, cause, ex, noise, held_noise = rng.standard_normal((6, n))
        x = u + 2 * ex
        a_index = u + 1.5 * inst + 0.2 * x + 0.3 * cause
        # Smooth expectation of A is enough for the population covariance check.
        a_mean = 0.1 + 0.8 * __import__("scipy").stats.norm.cdf(a_index)
        held = u + 0.9 * held_noise
        r_grid = np.linspace(-1, 1, 101)
        norms = []
        for r in r_grid:
            z = np.column_stack([x, cause, u + 0.9 * noise + r * inst])
            target = np.column_stack([inst, held])
            norms.append(np.linalg.norm(partial_covariance(a_mean, target, z)))
        self.assertGreater(min(norms), 0.01)

    def test_aggregate_retains_tied_largest_outputs(self):
        out = aggregate([(0,), (1,), (2,), (3,)], np.array([0.0, 0.01, 1.0, 1.01]), 0.01)
        self.assertEqual(out["return_kind"], "tied_largest")
        self.assertIsNone(out["output"])
        self.assertEqual(len(out["candidates"]), 2)

    def test_root_tie_rule_constant_is_frozen(self):
        self.assertEqual(ROOT_TIE_TOL, 1e-12)

    def test_restricted_brier_uses_independent_fit_and_select(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6)
        raw = generate_role(spec, 240, 91)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        selected = brier_select(view, (1,), 92, grid_points=5)
        self.assertEqual(selected["fit_n"] + selected["select_n"], 240)
        audit = brier_audit(view, (1,), selected)
        self.assertIn("signed_gap", audit)

    def test_result_writer_refuses_overwrite_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_atomic(path, {"complete": True})
            with self.assertRaises(FileExistsError):
                write_json_atomic(path, {"complete": False})
            bad = Path(directory) / "bad.json"
            with self.assertRaises(ValueError):
                write_json_atomic(bad, {"x": float("nan")})

    def test_array_hash_binds_shape_and_values(self):
        x = np.arange(6, dtype=np.float64)
        self.assertNotEqual(array_sha256(x), array_sha256(x.reshape(2, 3)))
        y = x.copy(); y[0] = 1
        self.assertNotEqual(array_sha256(x), array_sha256(y))


if __name__ == "__main__":
    unittest.main()
