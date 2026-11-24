# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""Unit tests of the hpimd 0.1.1 changes (AMENDMENT-HPIMD-0.2): run selection, the length
rule with its new fits flag, and the grader's merge of several jobs. Expectations are worked
out by hand in the comments. numpy and POSIX only (no torch, no model)."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import numpy  # noqa: F401
    import hpimd_sim as hs
    import grade_hpimd as gh
except ImportError:  # pragma: no cover
    hs = None

OK = hs is not None and os.name == "posix"


@unittest.skipUnless(OK, "needs numpy and POSIX")
class TestSelect(unittest.TestCase):
    def test_runs_unchanged_from_010(self):
        self.assertEqual(hs.RUNS["ctrl"], [("md", "Li2AuH6", 80.0, True),
                                           ("md", "Li2AgH6", 80.0, True),
                                           ("pimd", "Li2AuH6", 80.0, True),
                                           ("pimd", "Li2AgH6", 80.0, True)])
        self.assertEqual(hs.RUNS["tgt"], [("md", "Mg2IrH6", 80.0, True),
                                          ("pimd", "Mg2IrH6", 80.0, True),
                                          ("pimd", "Mg2IrH6", 300.0, False)])
        self.assertEqual(hs.SEED0, 20261122)
        self.assertEqual((hs.P_BEADS, hs.DT, hs.GAMMA0, hs.STRIDE), (16, 0.5, 0.01, 10))
        self.assertEqual((hs.MD_FS, hs.PIMD_CHOICES, hs.MIN_FS),
                         (1000.0, (1000.0, 800.0, 600.0, 500.0), 500.0))

    def test_select_keeps_indices(self):
        self.assertEqual(hs.select_runs("ctrl", "2"), [(2, ("pimd", "Li2AuH6", 80.0, True))])
        self.assertEqual([i for i, _ in hs.select_runs("ctrl", "3,1")], [1, 3])
        self.assertEqual(len(hs.select_runs("tgt", "all")), 3)
        self.assertEqual(len(hs.select_runs("tgt", "")), 3)

    def test_select_rejects(self):
        with self.assertRaises(ValueError):
            hs.select_runs("ctrl", "4")
        with self.assertRaises(ValueError):
            hs.select_runs("ctrl", "1,1")
        with self.assertRaises(ValueError):
            hs.select_runs("tgt", "-1")

    def test_versions(self):
        self.assertEqual(hs.VERSION, "0.1.1")
        self.assertEqual(gh.VERSION, "0.1.1")


@unittest.skipUnless(OK, "needs numpy and POSIX")
class TestPlan(unittest.TestCase):
    MD_IR = ("md", "Mg2IrH6", 80.0, True)
    P80 = ("pimd", "Mg2IrH6", 80.0, True)
    P300 = ("pimd", "Mg2IrH6", 300.0, False)

    def test_md_plus_pimd_shortens(self):
        # md 2000 x 1 = 2000 s; 1.0 ps: 2000 + 2000*8 = 18000 > 14450; 0.8: 2000 + 1600*8 =
        # 14800 > 14450; 0.6: 2000 + 1200*8 = 11600 <= 14450 -> 600 fs; fits: 2000 + 8000.
        p = hs.plan_lengths(1.0, 8.0, 14450.0, [self.MD_IR, self.P80])
        self.assertEqual(p["pimd_fs"], 600.0)
        self.assertTrue(p["fits"])
        self.assertAlmostEqual(p["estimate_s"], 11600.0)

    def test_too_slow_does_not_fit(self):
        # 0.5 ps: 2000 + 1000*20 = 22000 > 14450 -> fits False (job stops, exit 4)
        p = hs.plan_lengths(1.0, 20.0, 14450.0, [self.MD_IR, self.P80])
        self.assertEqual(p["pimd_fs"], 500.0)
        self.assertFalse(p["fits"])

    def test_md_only(self):
        # two MD runs, 2 x 2000 x 3 = 12000 <= 14450
        au = ("md", "Li2AuH6", 80.0, True)
        ag = ("md", "Li2AgH6", 80.0, True)
        p = hs.plan_lengths(3.0, 0.0, 14450.0, [au, ag])
        self.assertTrue(p["fits"])
        self.assertAlmostEqual(p["estimate_s"], 12000.0)
        self.assertFalse(hs.plan_lengths(4.0, 0.0, 14450.0, [au, ag])["fits"])  # 16000

    def test_secondary_dropped_first(self):
        # 1.0 ps primary: 2000*5 = 10000 <= 14450; with secondary 20000 > 14450 -> dropped
        p = hs.plan_lengths(0.0, 5.0, 14450.0, [self.P80, self.P300])
        self.assertEqual(p["pimd_fs"], 1000.0)
        self.assertFalse(p["include_secondary"])
        self.assertAlmostEqual(p["estimate_s"], 10000.0)

    def test_same_as_010_rule_for_full_target(self):
        # md 1000; 1.0 ps: 1000 + 2000*2 = 5000 <= 14450; with secondary 9000 <= 14450
        p = hs.plan_lengths(0.5, 2.0, 14450.0, [self.MD_IR, self.P80, self.P300])
        self.assertEqual(p["pimd_fs"], 1000.0)
        self.assertTrue(p["include_secondary"])
        self.assertAlmostEqual(p["estimate_s"], 9000.0)


@unittest.skipUnless(OK, "needs numpy and POSIX")
class TestMerge(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def _w(self, name, runs):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            json.dump({"runs": [{"tag": t, "class": c} for t, c in runs]}, fh)
        return p

    def test_three_ctrl_jobs_pass(self):
        f1 = self._w("md.json", [("md-Li2AuH6-80K", "INTACT"), ("md-Li2AgH6-80K", "HOST-COLLAPSE")])
        f2 = self._w("pau.json", [("pimd-Li2AuH6-80K", "H-BREAKDOWN")])
        f3 = self._w("pag.json", [("pimd-Li2AgH6-80K", "HOST-COLLAPSE")])
        t1 = self._w("tgt.json", [("md-Mg2IrH6-80K", "INTACT"), ("pimd-Mg2IrH6-80K", "INTACT")])
        gate, verdict, c, t = gh.grade_model(",".join([f1, f2, f3]), t1)
        self.assertEqual(gate["gate"], "PASS")
        self.assertEqual(verdict, "SURVIVES")
        self.assertEqual(len(c), 4)

    def test_missing_job_not_gradable(self):
        f1 = self._w("md.json", [("md-Li2AuH6-80K", "INTACT")])
        f2 = self._w("pau.json", [("pimd-Li2AuH6-80K", "H-BREAKDOWN")])
        gone = os.path.join(self.d, "absent.json")
        t1 = self._w("tgt.json", [("md-Mg2IrH6-80K", "INTACT"), ("pimd-Mg2IrH6-80K", "INTACT")])
        gate, verdict, _, _ = gh.grade_model(",".join([f1, f2, gone]), t1)
        self.assertEqual(gate["gate"], "NOT GRADABLE")
        self.assertEqual(verdict, "NOT GRADABLE (control gate NOT GRADABLE)")

    def test_no_files(self):
        gone = os.path.join(self.d, "absent.json")
        gate, verdict, c, t = gh.grade_model(gone, gone)
        self.assertIsNone(c)
        self.assertEqual(gate["gate"], "NOT GRADABLE")
        self.assertEqual(verdict, "NOT GRADABLE (no target job)")

    def test_duplicate_tag_fatal(self):
        f1 = self._w("a.json", [("md-Li2AuH6-80K", "INTACT")])
        f2 = self._w("b.json", [("md-Li2AuH6-80K", "PARTIAL")])
        with self.assertRaises(SystemExit):
            gh.classes(f1 + "," + f2)


if __name__ == "__main__":
    unittest.main()
