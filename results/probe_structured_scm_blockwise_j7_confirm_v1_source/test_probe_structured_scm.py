import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probe_structured_scm import (
    ALL_SPLITS,
    CAUSE_BLOCK,
    CONFOUNDER_BLOCK,
    N_BLOCKS,
    ROOT_TIE_TOL,
    SPLIT_CATEGORIES,
    budget_permutation,
    sample_splits,
    _brier_risk_gradient,
    SCMSpec,
    Transform,
    aggregate,
    average_score_vectors,
    array_sha256,
    baseline_score_arrays,
    brier_audit,
    brier_select,
    generate_role,
    heldout,
    observed_view,
    moment_root,
    partial_covariance,
    representation,
    run_e9_component_replicate,
    run_rotating_replicate,
    split_category,
    score_role_ledger,
    write_json_atomic,
)


class StructuredSCMTests(unittest.TestCase):
    def test_seven_blocks_enumerate_the_manuscript_split_set(self):
        """J = 7 single-coordinate blocks, so T = 2^J - 2 exactly as in Algorithm 1."""
        self.assertEqual(N_BLOCKS, 7)
        self.assertEqual(len(ALL_SPLITS), 2 ** N_BLOCKS - 2)
        counts = {key: 0 for key in SPLIT_CATEGORIES}
        for split in ALL_SPLITS:
            counts[split_category(split)] += 1
        self.assertEqual(
            counts,
            {"clean": 15, "bad_singleton": 1, "mixed": 15, "drops_cause": 32, "drops_confounder": 63},
        )

    def test_representation_uses_exactly_the_retained_blocks(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6)
        raw = generate_role(spec, 200, 5)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        # X, confounder, tilted channel when both special blocks are retained.
        self.assertEqual(representation(view, (0,), 0.3).shape[1], 3)
        # the confounder block held out removes its column
        self.assertEqual(representation(view, (CONFOUNDER_BLOCK,), 0.3).shape[1], 2)
        # the cause block held out removes the tilt, and then r must be absent
        self.assertEqual(representation(view, (CAUSE_BLOCK,), None).shape[1], 3)
        with self.assertRaises(ValueError):
            representation(view, (CAUSE_BLOCK,), 0.3)
        with self.assertRaises(ValueError):
            representation(view, (0,), None)

    def test_heldout_returns_one_column_per_block(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6)
        raw = generate_role(spec, 120, 6)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        for split in ((0,), (3, CAUSE_BLOCK), (0, 1, CONFOUNDER_BLOCK)):
            self.assertEqual(heldout(view, split).shape, (120, len(split)))

    def test_budget_draws_are_nested_and_without_replacement(self):
        order = budget_permutation(31)
        self.assertEqual(len(set(order)), len(ALL_SPLITS))
        self.assertEqual(sample_splits(10, 31), order[:10])
        self.assertEqual(sample_splits(30, 31)[:10], order[:10])
        self.assertEqual(sample_splits(None, 31), ALL_SPLITS)

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

    def test_restricted_brier_uses_same_D_empirical_objective(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6)
        raw = generate_role(spec, 240, 91)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        selected = brier_select(view, (1,), 92, grid_points=5)
        self.assertEqual(selected["empirical_n"], 240)
        self.assertGreaterEqual(selected["signed_empirical_gap"], -1e-9)
        self.assertIn(selected["critic1_fit"]["chosen_source"], ("optimized", "unoptimized_start"))

    def test_brier_selection_is_deterministic_given_the_same_D(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6)
        raw = generate_role(spec, 180, 191)
        view = observed_view(raw, Transform.fit(raw, "linear"))
        left = brier_select(view, (1,), 1, grid_points=5)
        right = brier_select(view, (1,), 999999, grid_points=5)
        self.assertEqual(left["r"], right["r"])
        self.assertEqual(left["objective"], right["objective"])
        np.testing.assert_array_equal(left["beta0"], right["beta0"])
        np.testing.assert_array_equal(left["beta1"], right["beta1"])

    def test_score_role_ledger_rejects_overlap(self):
        row = score_role_ledger((0, 1), (1, 0), 2, 3, 1500)
        self.assertEqual(row["unique_evaluation_rows"], 1500)
        with self.assertRaises(ValueError):
            score_role_ledger((0,), (1,), 2, 1, 1500)

    def test_average_scores_keeps_one_row_per_evaluation_unit(self):
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([3.0, 4.0, 5.0])
        out = average_score_vectors([left, right])
        np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0]))
        self.assertEqual(len(out), len(left))
        with self.assertRaises(ValueError):
            average_score_vectors([left, right[:2]])

    def test_shared_baselines_use_exact_supplied_nuisance_and_evaluation_rows(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6)
        nuisance = generate_role(spec, 120, 202)
        evaluation = generate_role(spec, 80, 203)
        transform = Transform.fit(generate_role(spec, 100, 201), "linear")
        scores = baseline_score_arrays(nuisance, evaluation, transform)
        self.assertEqual(set(scores), {"X", "raw_XW", "oracle"})
        self.assertTrue(all(len(value) == 80 for value in scores.values()))

    def test_component_ablation_has_honest_unique_row_accounting(self):
        spec = SCMSpec("test", 2.5, 0.85, 2.5, -0.6, outcome_noise="constant_small")
        run = run_e9_component_replicate(spec, 240, 1200, grid_points=3)
        expected = {"H11": 60, "H12": 120, "H21": 60, "H22": 120}
        self.assertEqual(set(run["arms"]), set(expected))
        for name, arm in run["arms"].items():
            self.assertEqual(arm["unique_evaluation_rows"], expected[name])
            self.assertTrue(all(row["unique_evaluation_rows"] == 60 for row in arm["score_ledgers"]))
            for ledger in arm["score_ledgers"]:
                training = set(ledger["representation_folds"] + ledger["screen_folds"])
                training.add(ledger["nuisance_fold"])
                self.assertNotIn(ledger["evaluation_fold"], training)
        self.assertEqual(run["arms"]["H21"]["ds_pairs"], [{"D": 0, "S": 1}, {"D": 1, "S": 0}])

    @unittest.skip(
        "pinned to the five-block bundled partition; replay it against the sealed source "
        "snapshot results/probe_structured_scm_brier_g_confirm_v2_source, not this file"
    )
    def test_rotate4_refactor_reproduces_stored_e9_seed(self):
        path = Path(__file__).resolve().parents[1] / "results" / "probe_e9_scm1_noise_10_rotate4.json"
        stored = json.loads(path.read_text())["rows"][0]
        replay = run_rotating_replicate(
            SCMSpec(**stored["spec"]), stored["total_n"], stored["fold_seeds"][0], learner="brier"
        )
        self.assertEqual(replay["fold_sha256"], stored["fold_sha256"])
        self.assertEqual(replay["aggregation"], stored["aggregation"])
        self.assertEqual(replay["baseline_estimates"], stored["baseline_estimates"])

    def test_component_cli_fails_closed_on_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.json"
            output = Path(directory) / "output.json"
            protocol.write_text(json.dumps({
                "protocol_id": "e9-hard4-component-ablation-v1",
                "source_sha256": {"probe_structured_scm.py": "not-a-real-hash"},
            }))
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).with_name("run_e9_component_ablation.py")),
                 "--protocol", str(protocol), "--output", str(output)],
                text=True, capture_output=True, check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source hash differs", result.stderr + result.stdout)
            self.assertFalse(output.exists())

    def test_brier_analytic_gradient_matches_directional_difference(self):
        rng = np.random.default_rng(616)
        z = rng.normal(size=(80, 3)); a = rng.binomial(1, 0.5, size=80)
        beta = rng.normal(scale=0.2, size=4); direction = rng.normal(size=4)
        _, gradient = _brier_risk_gradient(beta, z, a)
        eps = 1e-6
        plus = _brier_risk_gradient(beta + eps * direction, z, a)[0]
        minus = _brier_risk_gradient(beta - eps * direction, z, a)[0]
        numerical = (plus - minus) / (2 * eps)
        self.assertAlmostEqual(float(gradient @ direction), numerical, places=7)

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

    def test_low_outcome_noise_changes_variance_not_generated_mean(self):
        standard = SCMSpec("s", 2.5, 0.85, 2.5, -0.6)
        small = SCMSpec("l", 2.5, 0.85, 2.5, -0.6, outcome_noise="constant_small")
        a = generate_role(standard, 1000, 303)
        b = generate_role(small, 1000, 303)
        for key in ("X", "I", "C", "M", "A", "U_eval_only"):
            np.testing.assert_array_equal(a[key], b[key])
        mean = a["A"] - 2.5 * a["U_eval_only"] + 0.2 * a["X"] - 0.5 * a["C"]
        np.testing.assert_allclose(b["Y"] - mean, 0.3 * (a["Y"] - mean))


if __name__ == "__main__":
    unittest.main()


class TestNestedSamples(unittest.TestCase):
    """A sample size curve has to grow one sample, not draw a new one at every size.

    Regenerating a role at a fixed seed does not nest.  U, X, I, C and the proxies happen to agree on the
    prefix because each is drawn as one whole-array call, but the treatment draw consumes a variable
    number of stream values and the outcome noise then starts at a different offset.
    """

    def setUp(self):
        from probe_structured_scm import SCM_FAMILY
        self.spec = SCM_FAMILY["SCM-1"]

    def test_regenerating_at_a_fixed_seed_breaks_the_treatment_and_outcome(self):
        from probe_structured_scm import generate_role
        small = generate_role(self.spec, 375, 91000000)
        large = generate_role(self.spec, 3000, 91000000)
        for field in ("U_eval_only", "X", "I", "C", "M", "p_eval_only"):
            np.testing.assert_array_equal(np.asarray(small[field]),
                                          np.asarray(large[field])[:375], field)
        for field in ("A", "Y"):
            self.assertFalse(np.array_equal(np.asarray(small[field]),
                                            np.asarray(large[field])[:375]), field)

    def test_slicing_one_draw_nests_every_field(self):
        from probe_structured_scm import generate_role, slice_role
        big = generate_role(self.spec, 3000, 91000000)
        for rows in (375, 750, 1500):
            cut = slice_role(big, rows)
            for field in big:
                np.testing.assert_array_equal(np.asarray(cut[field]),
                                              np.asarray(big[field])[:rows], field)

    def test_slicing_refuses_to_invent_rows(self):
        from probe_structured_scm import generate_role, slice_role
        with self.assertRaises(ValueError):
            slice_role(generate_role(self.spec, 400, 1), 500)

    def test_supplied_folds_must_match_the_requested_size(self):
        from probe_structured_scm import generate_role, run_rotating_replicate
        folds = [generate_role(self.spec, 375, 91000000 + j) for j in range(4)]
        with self.assertRaises(ValueError):
            run_rotating_replicate(self.spec, 3000, 91000000, learner="brier", folds=folds)
