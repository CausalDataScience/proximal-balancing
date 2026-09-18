"""Roles, folds and the write guards that keep a frozen protocol honest."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import minimal_suite_common as c


class TestFoldsAndRoles(unittest.TestCase):
    def test_four_folds_partition_every_row_exactly_once(self):
        folds = c.four_folds(12000, 11)
        self.assertEqual(len(folds), 4)
        joined = np.concatenate(folds)
        self.assertEqual(len(joined), 12000)
        np.testing.assert_array_equal(np.sort(joined), np.arange(12000))

    def test_four_folds_are_reproducible_from_the_seed(self):
        a = c.four_folds(8000, 5)
        b = c.four_folds(8000, 5)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], c.four_folds(8000, 6)[0]))

    def test_every_rotation_gives_four_disjoint_roles(self):
        folds = c.four_folds(4000, 3)
        seen = set()
        for rot in c.rotations(folds):
            self.assertEqual(sorted(rot.roles), ["D", "E", "N", "S"])
            report = c.assert_roles_disjoint(rot.roles)
            self.assertTrue(all(v == 0 for v in report.values()), report)
            seen.add(tuple(sorted(rot.roles["D"][:3])))
        self.assertEqual(len(seen), 4, "each rotation should put a different fold in D")

    def test_overlapping_roles_are_rejected(self):
        roles = {"D": np.arange(10), "S": np.arange(5, 15),
                 "N": np.arange(20, 30), "E": np.arange(30, 40)}
        with self.assertRaises(Exception):
            c.assert_roles_disjoint(roles)

    def test_nested_prefix_is_actually_nested(self):
        folds = c.four_folds(9000, 2)
        small = c.nested_prefix(folds, 500)
        large = c.nested_prefix(folds, 1500)
        for s, l in zip(small, large):
            np.testing.assert_array_equal(s, l[:len(s)])

    def test_nested_prefix_refuses_more_rows_than_a_fold_holds(self):
        with self.assertRaises(ValueError):
            c.nested_prefix(c.four_folds(400, 1), 500)


class TestSplits(unittest.TestCase):
    def test_thirty_oriented_splits_none_removed_by_truth(self):
        splits = c.oriented_splits()
        self.assertEqual(len(splits), 30)
        self.assertEqual(len(set(splits)), 30)
        self.assertNotIn((), splits)
        self.assertNotIn((0, 1, 2, 3, 4), splits)

    def test_evaluator_category_never_reaches_the_split_identity(self):
        for split in c.oriented_splits():
            self.assertIn(c.evaluator_category(split),
                          {"block0_heldout", "bad_singleton", "mixed", "clean"})
            self.assertTrue(c.split_id(split).startswith("S"))

    def test_optimizer_seed_depends_on_every_argument(self):
        base = c.optimizer_seed("p", 1, 0, "S1", 0)
        self.assertNotEqual(base, c.optimizer_seed("q", 1, 0, "S1", 0))
        self.assertNotEqual(base, c.optimizer_seed("p", 2, 0, "S1", 0))
        self.assertNotEqual(base, c.optimizer_seed("p", 1, 1, "S1", 0))
        self.assertNotEqual(base, c.optimizer_seed("p", 1, 0, "S2", 0))
        self.assertNotEqual(base, c.optimizer_seed("p", 1, 0, "S1", 1))
        self.assertEqual(base, c.optimizer_seed("p", 1, 0, "S1", 0))


class TestWriteGuards(unittest.TestCase):
    def test_write_json_new_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            c.write_json_new(path, {"a": 1})
            with self.assertRaises(Exception):
                c.write_json_new(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text())["a"], 1)

    def test_write_json_new_rejects_a_nan(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            with self.assertRaises(ValueError):
                c.write_json_new(path, {"a": float("nan")})
            self.assertFalse(path.exists(), "a rejected payload must leave no file behind")

    def test_source_snapshot_digests_the_real_file_and_the_bundle_covers_all_of_them(self):
        here = Path(c.__file__).parent
        one = c.source_snapshot(["minimal_suite_common.py"])
        self.assertEqual(one["files"]["minimal_suite_common.py"],
                         c.file_sha256(here / "minimal_suite_common.py"))
        two = c.source_snapshot(["minimal_suite_common.py", "probe_highdim_w.py"])
        self.assertNotEqual(one["bundle_sha256"], two["bundle_sha256"],
                            "the bundle digest must move when the file set changes")
        self.assertEqual(two["files"]["minimal_suite_common.py"],
                         one["files"]["minimal_suite_common.py"])

    def test_check_protocol_rejects_a_source_that_no_longer_hashes_as_frozen(self):
        protocol = {"runs": [{"experiment_id": "SCM-6", "mode": "operational_rotate4",
                              "phase": "development", "seeds": [1, 2]}],
                    "source_snapshot": {"files": {"minimal_suite_common.py": "0" * 64}}}
        with self.assertRaises(SystemExit):
            c.check_protocol(protocol, protocol["runs"][0], "nowhere.json",
                             ["minimal_suite_common.py"])

    def test_check_protocol_rejects_a_run_the_protocol_does_not_declare(self):
        protocol = {"runs": [{"experiment_id": "SCM-6", "mode": "operational_rotate4",
                              "phase": "development", "seeds": [1, 2]}],
                    "source_snapshot": c.source_snapshot(["minimal_suite_common.py"])}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "protocol.json"
            sealed = c.seal_protocol(path, protocol)
            with self.assertRaises(SystemExit):
                c.check_protocol(sealed, {"experiment_id": "SCM-6", "mode": "operational_rotate4",
                                          "phase": "confirmation", "seeds": [1, 2]},
                                 str(path), ["minimal_suite_common.py"])
            c.check_protocol(sealed, sealed["runs"][0], str(path), ["minimal_suite_common.py"])

    def test_an_unsealed_protocol_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "protocol.json"
            protocol = {"runs": [{"phase": "development"}], "source_snapshot": {"files": {}}}
            path.write_text(json.dumps(protocol))
            with self.assertRaises(SystemExit):
                c.check_protocol(protocol, {"phase": "development"}, str(path), [])

    def test_a_sealed_protocol_passes_and_any_later_edit_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "protocol.json"
            sealed = c.seal_protocol(path, {"runs": [{"phase": "development"}], "n": 12000,
                                            "source_snapshot": {"files": {}}})
            c.check_protocol(sealed, {"phase": "development"}, str(path), [])
            for tampered in ({**sealed, "n": 24000},
                             {**sealed, "runs": [{"phase": "confirmation"}]},
                             {**sealed, c.DIGEST_FIELD: "0" * 64}):
                with self.assertRaises(SystemExit):
                    c.check_protocol(tampered, {"phase": "development"}, str(path), [])

    def test_the_digest_ignores_key_order_and_only_that_one_field(self):
        base = {"b": 2, "a": 1, "runs": []}
        self.assertEqual(c.canonical_digest(base), c.canonical_digest({"a": 1, "runs": [], "b": 2}))
        self.assertEqual(c.canonical_digest(base),
                         c.canonical_digest({**base, c.DIGEST_FIELD: "anything"}))
        self.assertNotEqual(c.canonical_digest(base), c.canonical_digest({**base, "a": 2}))

    def test_sealing_writes_once(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "protocol.json"
            c.seal_protocol(path, {"runs": []})
            with self.assertRaises(FileExistsError):
                c.seal_protocol(path, {"runs": []})


if __name__ == "__main__":
    unittest.main()
