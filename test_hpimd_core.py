# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""Unit tests of hpimd_core (numpy only). Expectations worked out by hand in the comments."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import numpy as np
    import hpimd_core as hc
except ImportError:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy not available")
class TestNormalModes(unittest.TestCase):
    def test_orthogonal(self):
        for P in (1, 2, 5, 16):
            C = hc.normal_mode_matrix(P)
            self.assertTrue(np.allclose(C.T @ C, np.eye(P), atol=1e-12))

    def test_diagonalises_ring(self):
        # spring energy (1/2) m wP^2 sum_j (q_j - q_{j+1})^2 = (1/2) m sum_k w_k^2 qt_k^2
        for P in (2, 6, 16):
            L = 2 * np.eye(P) - np.roll(np.eye(P), 1, axis=1) - np.roll(np.eye(P), -1, axis=1)
            if P == 2:
                L = np.array([[2.0, -2.0], [-2.0, 2.0]])
            C = hc.normal_mode_matrix(P)
            T = 100.0
            wP = P * hc.KB * T / hc.HBAR
            M = C.T @ (wP ** 2 * L) @ C
            self.assertTrue(np.allclose(M, np.diag(hc.ring_frequencies(P, T) ** 2), atol=1e-10))

    def test_exact_limits(self):
        # P = 1: classical kT/(m w^2); large P: hbar/(2 m w) coth(beta hbar w / 2)
        m, w, T = 1.0, 0.1, 191.0
        cl = hc.KB * T * hc.ACC / (m * w ** 2)
        self.assertAlmostEqual(hc.harmonic_x2_exact(1, T, m, w) / cl, 1.0, places=12)
        bhw = hc.HBAR * w / (hc.KB * T)
        qm = hc.HBAR * hc.ACC / (2 * m * w) / np.tanh(bhw / 2)
        self.assertAlmostEqual(hc.harmonic_x2_exact(512, T, m, w) / qm, 1.0, places=4)


@unittest.skipIf(np is None, "numpy not available")
class TestIntegrator(unittest.TestCase):
    def test_free_ring_energy_conserved(self):
        # no forces, no thermostat: the free ring is propagated exactly
        P, T, N = 8, 80.0, 3
        m = np.array([1.008, 6.94, 196.97])
        rp = hc.RingPolymerLangevin(np.zeros((N, 3)), m, P, T, 0.5, 0.0,
                                    lambda q: (np.zeros_like(q), np.zeros(q.shape[0])), seed=1)
        rp.c1[:] = 1.0
        rp.c2[:] = 0.0
        rp.q += 0.05 * np.random.default_rng(2).standard_normal(rp.q.shape)

        def energy():
            wP = P * hc.KB * T / hc.HBAR
            ke = 0.5 * np.sum(m[None, :, None] * rp.v ** 2) / hc.ACC
            sp = 0.5 * wP ** 2 * np.sum(m[None, :, None] * (rp.q - np.roll(rp.q, -1, 0)) ** 2) / hc.ACC
            return ke + sp
        e0 = energy()
        for _ in range(500):
            rp.step()
        self.assertAlmostEqual(energy() / e0, 1.0, places=9)

    def _harmonic(self, P):
        m, w = 1.0, 0.1                      # amu, 1/fs
        T = hc.HBAR * w / (4.0 * hc.KB)      # beta hbar w = 4 (about 191 K): strongly quantum
        N = 400
        k = m * w ** 2 / hc.ACC              # eV / A^2

        def ff(q):
            return -k * q, 0.5 * k * np.sum(q ** 2, axis=(1, 2))
        rp = hc.RingPolymerLangevin(np.zeros((N, 3)), np.full(N, m), P, T, 0.5, 0.05, ff, seed=7)
        acc_b, acc_c, temps = [], [], []
        for s in range(6000):
            rp.step()
            if s >= 1000 and s % 5 == 0:
                acc_b.append(np.mean(rp.q ** 2))
                acc_c.append(np.mean(rp.centroid() ** 2))
                temps.append(rp.kinetic_T())
        return m, w, T, np.mean(acc_b), np.mean(acc_c), np.mean(temps)

    def test_harmonic_quantum(self):
        # bead <x^2> must match the exact P = 16 value (2.07 x classical here) within 3 %;
        # centroid variance is exactly classical kT/(m w^2) for a harmonic well
        m, w, T, xb, xc, tk = self._harmonic(16)
        exact = hc.harmonic_x2_exact(16, T, m, w)
        cl = hc.harmonic_x2_exact(1, T, m, w)
        self.assertGreater(exact / cl, 1.9)
        self.assertAlmostEqual(xb / exact, 1.0, delta=0.03)
        self.assertAlmostEqual(xc / cl, 1.0, delta=0.05)
        self.assertAlmostEqual(tk / T, 1.0, delta=0.03)

    def test_harmonic_classical(self):
        m, w, T, xb, xc, tk = self._harmonic(1)
        cl = hc.harmonic_x2_exact(1, T, m, w)
        self.assertAlmostEqual(xb / cl, 1.0, delta=0.03)
        self.assertAlmostEqual(tk / T, 1.0, delta=0.03)


@unittest.skipIf(np is None, "numpy not available")
class TestStructures(unittest.TestCase):
    def test_counts_and_distances(self):
        a, x = 6.66, 0.26
        sym, pos, cell = hc.a2bh6("Mg", "Ir", a, x, n=2)
        self.assertEqual(len(sym), 72)
        self.assertEqual(sym.count("H"), 48)
        self.assertEqual(sym.count("Mg"), 16)
        self.assertEqual(sym.count("Ir"), 8)
        self.assertAlmostEqual(abs(np.linalg.det(cell)), 8 * a ** 3 / 4, places=6)
        D = np.linalg.norm(hc.mic(pos[:, None] - pos[None], cell), axis=2)
        np.fill_diagonal(D, np.inf)
        s = np.array(sym)
        ir, h, mg = s == "Ir", s == "H", s == "Mg"
        self.assertAlmostEqual(D[np.ix_(ir, h)].min(), x * a, places=9)          # Ir-H 1.7316
        # nearest H-H: sqrt(2) a min(x, 1/2 - x) = sqrt(2) * 0.24 * 6.66 = 2.2605
        self.assertAlmostEqual(D[np.ix_(h, h)].min(), np.sqrt(2) * a * 0.24, places=9)
        # Mg-Ir: a sqrt(3)/4 ; every Ir has exactly 6 H within 1.9 A
        self.assertAlmostEqual(D[np.ix_(mg, ir)].min(), a * np.sqrt(3) / 4, places=9)
        self.assertTrue(np.all((D[np.ix_(ir, h)] < 1.9).sum(axis=1) == 6))

    def test_parabola(self):
        xs = np.linspace(0.2, 0.3, 11)
        es = 3.0 * (xs - 0.2537) ** 2
        xv, edge = hc.parabola_min(xs, es)
        self.assertFalse(edge)
        self.assertAlmostEqual(xv, 0.2537, places=9)
        self.assertTrue(hc.parabola_min(xs, -xs)[1])


@unittest.skipIf(np is None, "numpy not available")
class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.sym, self.r0, self.cell = hc.a2bh6("Li", "Au", 4.71 * np.sqrt(2), 0.25, n=2)
        self.m = np.array([hc.MASS[s] for s in self.sym])

    def test_ideal_frame(self):
        fm = hc.frame_metrics(self.r0, self.r0, self.sym, self.m, self.cell)
        self.assertEqual(fm["dimer"], 0.0)
        self.assertEqual(fm["wander"], 0.0)
        self.assertEqual(fm["host_out"], 0.0)
        self.assertEqual(fm["clusters"], 1.0)
        self.assertEqual(hc.classify(fm["dimer"], fm["wander"], fm["host_out"], False), "INTACT")

    def test_one_dimer(self):
        # pull two cis H of the first Au together to 0.74 A (an H2 molecule), symmetric about
        # their midpoint (x a / 2)(1, 1, 0) = (0.833, 0.833, 0) A
        r = self.r0.copy()
        h = [i for i, s in enumerate(self.sym) if s == "H"]
        i, j = h[0], h[2]                         # (+x,0,0) and (0,+x,0) of the first Au
        mid = 0.5 * (r[i] + r[j])
        u = (r[i] - r[j]) / np.linalg.norm(r[i] - r[j])
        r[i], r[j] = mid + 0.37 * u, mid - 0.37 * u
        fm = hc.frame_metrics(r, self.r0, self.sym, self.m, self.cell)
        self.assertAlmostEqual(fm["dimer"], 2 / 48)
        self.assertAlmostEqual(fm["min_hh"], 0.74, places=9)
        # each moved H travelled |r - r0| = 2.355/2 - 0.37 ~ 0.81 A < 1 A: not wandered
        self.assertEqual(fm["wander"], 0.0)
        # the moved H sit at (1.095, 0.571, 0) and (0.571, 1.095, 0): 1.235 A from their Au, so
        # all 8 AuH6 clusters still count as intact (6 H within 1.9 A)
        self.assertAlmostEqual(fm["clusters"], 1.0)
        self.assertEqual(hc.classify(fm["dimer"], fm["wander"], fm["host_out"], False), "PARTIAL")

    def test_classes(self):
        self.assertEqual(hc.classify(0.0, 0.0, 0.3, False), "HOST-COLLAPSE")
        self.assertEqual(hc.classify(0.236, 0.1, 0.0, False), "H-BREAKDOWN")
        self.assertEqual(hc.classify(0.0, 0.25, 0.1, False), "H-BREAKDOWN")
        self.assertEqual(hc.classify(0.05, 0.0, 0.0, False), "PARTIAL")
        self.assertEqual(hc.classify(0.0, 0.0, 0.0, True), "UNPHYSICAL")

    def test_summarize_window_and_length(self):
        fr = [{"dimer": 0.0, "wander": 0.0, "host_out": 0.0, "h_disp": 0.1, "host_disp": 0.05,
               "min_hh": 2.0, "min_heavy": 1.5, "clusters": 1.0}] * 5
        fr2 = [dict(f, dimer=0.3) for f in fr]
        s = hc.summarize(fr + fr2, np.arange(10) * 100.0, 1000.0)
        self.assertAlmostEqual(s["dimer"], 0.3)            # window = t >= 500 fs: frames 5..9
        self.assertEqual(s["class"], "H-BREAKDOWN")
        self.assertEqual(hc.summarize(fr, np.arange(5) * 100.0, 400.0)["class"], "NOT GRADABLE")
        bad = [dict(fr[0], min_hh=0.3)]
        self.assertEqual(hc.summarize(fr + bad, np.arange(6) * 100.0, 600.0)["class"], "UNPHYSICAL")

    def test_gate_and_verdicts(self):
        g = hc.control_gate("INTACT", "H-BREAKDOWN", "HOST-COLLAPSE")
        self.assertEqual(g["gate"], "PASS")
        self.assertEqual(hc.control_gate("H-BREAKDOWN", "H-BREAKDOWN", "HOST-COLLAPSE")["gate"], "FAIL")
        self.assertEqual(hc.control_gate("INTACT", "INTACT", "HOST-COLLAPSE")["gate"], "FAIL")
        self.assertEqual(hc.control_gate("INTACT", "H-BREAKDOWN", "INTACT")["gate"], "FAIL")
        self.assertEqual(hc.control_gate("INTACT", "H-BREAKDOWN", "NOT GRADABLE")["gate"],
                         "NOT GRADABLE")
        self.assertEqual(hc.target_verdict("PASS", "INTACT", "INTACT"), "SURVIVES")
        self.assertEqual(hc.target_verdict("PASS", "INTACT", "H-BREAKDOWN"), "KINETICALLY UNSTABLE")
        self.assertTrue(hc.target_verdict("FAIL", "INTACT", "INTACT").startswith("NOT GRADABLE"))
        self.assertTrue(hc.target_verdict("PASS", "HOST-COLLAPSE", "INTACT").startswith("NOT GRADABLE"))
        self.assertEqual(hc.combine(["SURVIVES", "SURVIVES"]), "SURVIVES (two models)")
        self.assertEqual(hc.combine(["SURVIVES", "NOT GRADABLE (x)"]), "SURVIVES (one model)")
        self.assertTrue(hc.combine(["SURVIVES", "PARTIAL"]).startswith("INCONCLUSIVE"))
        self.assertEqual(hc.combine(["NOT GRADABLE (a)", "NOT GRADABLE (b)"]), "NOT GRADABLE")


if __name__ == "__main__":
    unittest.main()
