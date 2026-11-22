# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""hpimd_core 0.1.0 -- numpy-only core of the hydride quantum-stability test (PREREG-v0.1.md).

Contents
* ring-polymer molecular dynamics (PIMD) with the PILE-L thermostat of Ceriotti, Parrinello,
  Markland and Manolopoulos, J. Chem. Phys. 133, 124104 (2010): exact free ring-polymer
  propagation in normal modes, critically damped internal modes (gamma_k = 2 omega_k) and a
  Langevin centroid with friction gamma0. P = 1 is ordinary Langevin MD.
* the cubic Fm-3m A2BH6 (K2PtCl6-type) structures of Li2AgH6, Li2AuH6 and Mg2IrH6;
* the pre-registered stability metrics (dimer fraction D, hydrogen wander W, host out-of-site
  fraction F, unphysical-contact flags) and the run classes and gates of PREREG-v0.1 sec. 4-5.

Units: energy eV, length Angstrom, mass amu, time fs, temperature K. Velocities in Angstrom/fs.
"""
import numpy as np

VERSION = "0.1.0"
HBAR = 0.6582119569        # eV fs
KB = 8.617333262e-5        # eV / K
ACC = 9.6485332e-3         # 1 eV/(Angstrom amu) expressed in Angstrom/fs^2
MASS = {"H": 1.008, "Li": 6.94, "Mg": 24.305, "Ag": 107.8682, "Au": 196.96657, "Ir": 192.217}

# Published DFT (PBE) lattice constants; conventional cubic a in Angstrom.
SYSTEMS = {
    "Li2AgH6": {"A": "Li", "B": "Ag", "a": 4.62 * np.sqrt(2.0),
                "source": "Ding, Chen, Shi arXiv:2604.12367: primitive 4.62 A (PBE, VASP)"},
    "Li2AuH6": {"A": "Li", "B": "Au", "a": 4.71 * np.sqrt(2.0),
                "source": "Ding, Chen, Shi arXiv:2604.12367: primitive 4.71 A (PBE, VASP)"},
    "Mg2IrH6": {"A": "Mg", "B": "Ir", "a": 6.66,
                "source": "Dolui et al. PRL 132, 166001 (2024), arXiv:2310.07562: a = 6.66 A"},
}

# Pre-registered thresholds (PREREG-v0.1 sec. 4). Never change them after data.
DIMER_R = 1.0          # A: H with another H centroid closer than this is "dimerised"
WANDER_R = 1.0         # A: centroid farther than this from its starting site has left it
CLUSTER_R = 1.9        # A: B-H distance defining an intact BH6 cluster (Dolui et al.)
MIN_HH = 0.45          # A: any H-H centroid pair closer -> UNPHYSICAL
MIN_HEAVY = 0.90       # A: any pair involving a non-H atom closer -> UNPHYSICAL
FMAX = 100.0           # eV/A: any force component larger (or non-finite) -> UNPHYSICAL


# ----------------------------------------------------------------------------- ring polymer
def normal_mode_matrix(P):
    """Orthogonal P x P matrix C with q_j = sum_k C[j,k] qt_k (free ring-polymer normal modes)."""
    C = np.zeros((P, P))
    j = np.arange(P)
    for k in range(P):
        if k == 0:
            C[:, k] = 1.0 / np.sqrt(P)
        elif 2 * k < P:
            C[:, k] = np.sqrt(2.0 / P) * np.cos(2 * np.pi * j * k / P)
        elif 2 * k == P:
            C[:, k] = (-1.0) ** j / np.sqrt(P)
        else:
            C[:, k] = np.sqrt(2.0 / P) * np.sin(2 * np.pi * j * k / P)
    return C


def ring_frequencies(P, T):
    """omega_k = 2 omega_P sin(pi k / P), omega_P = P kB T / hbar (1/fs)."""
    wP = P * KB * T / HBAR
    return 2.0 * wP * np.sin(np.pi * np.arange(P) / P)


def harmonic_x2_exact(P, T, m, w):
    """Exact <x^2> per Cartesian component of a P-bead ring polymer in V = m w^2 x^2 / 2
    (bead average): (kB T ACC / m) * sum_k 1/(w^2 + w_k^2)."""
    wk = ring_frequencies(P, T)
    return KB * T * ACC / m * np.sum(1.0 / (w ** 2 + wk ** 2))


class RingPolymerLangevin:
    """PILE-L ring-polymer integrator. force_fn(q[P,N,3]) -> (forces[P,N,3] eV/A, energies[P] eV)."""

    def __init__(self, q0, masses, P, T, dt, gamma0, force_fn, seed):
        self.P, self.T, self.dt = int(P), float(T), float(dt)
        self.m = np.asarray(masses, float)
        self.rng = np.random.default_rng(seed)
        self.C = normal_mode_matrix(self.P)
        self.w = ring_frequencies(self.P, self.T)
        self.sig = np.sqrt(KB * self.P * self.T * ACC / self.m)      # bead velocity std at T_P
        gam = 2.0 * self.w
        gam[0] = gamma0
        self.c1 = np.exp(-0.5 * self.dt * gam)
        self.c2 = np.sqrt(1.0 - self.c1 ** 2)
        wdt = self.w * self.dt
        self.cw = np.cos(wdt)
        safe = np.where(self.w > 0, self.w, 1.0)
        self.swow = np.where(self.w > 0, np.sin(wdt) / safe, self.dt)
        self.wsw = self.w * np.sin(wdt)
        self.q = np.repeat(np.asarray(q0, float)[None], self.P, axis=0)
        vt = self.sig[None, :, None] * self.rng.standard_normal(self.q.shape)
        self.v = self._from_nm(vt)
        self.force_fn = force_fn
        self.f, self.e = force_fn(self.q)
        self.nstep = 0

    def _to_nm(self, x):
        return np.einsum("jk,jna->kna", self.C, x)

    def _from_nm(self, x):
        return np.einsum("jk,kna->jna", self.C, x)

    def _thermo(self):
        vt = self._to_nm(self.v)
        vt = (self.c1[:, None, None] * vt
              + self.c2[:, None, None] * self.sig[None, :, None] * self.rng.standard_normal(vt.shape))
        self.v = self._from_nm(vt)

    def _free(self):
        qt, vt = self._to_nm(self.q), self._to_nm(self.v)
        cw, s, ws = self.cw[:, None, None], self.swow[:, None, None], self.wsw[:, None, None]
        self.q = self._from_nm(cw * qt + s * vt)
        self.v = self._from_nm(-ws * qt + cw * vt)

    def _kick(self):
        self.v += 0.5 * self.dt * ACC * self.f / self.m[None, :, None]

    def step(self):
        self._thermo()
        self._kick()
        self._free()
        self.f, self.e = self.force_fn(self.q)
        self._kick()
        self._thermo()
        self.nstep += 1

    def centroid(self):
        return self.q.mean(axis=0)

    def kinetic_T(self):
        ke = 0.5 * np.sum(self.m[None, :, None] * self.v ** 2) / ACC
        return 2.0 * ke / (3.0 * self.m.size * self.P * KB) / self.P

    def forces_ok(self):
        return bool(np.all(np.isfinite(self.f)) and np.max(np.abs(self.f)) <= FMAX
                    and np.all(np.isfinite(self.e)))


# ----------------------------------------------------------------------------- structures
def a2bh6(A, B, a, x, n=2):
    """Fm-3m A2BH6: B at 4a (0,0,0), A at 8c (1/4,1/4,1/4), H at 24e (x,0,0); fcc primitive cell
    repeated n x n x n (n = 2 -> 72 atoms). Returns (symbols, positions[N,3], cell[3,3])."""
    v = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    basis = [(B, (0, 0, 0)), (A, (0.25, 0.25, 0.25)), (A, (0.75, 0.75, 0.75))]
    for s in [(x, 0, 0), (-x, 0, 0), (0, x, 0), (0, -x, 0), (0, 0, x), (0, 0, -x)]:
        basis.append(("H", s))
    symbols, pos = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t = i * v[0] + j * v[1] + k * v[2]
                for s, f in basis:
                    symbols.append(s)
                    pos.append(t + a * np.array(f, float))
    return symbols, np.array(pos), n * v


def wrap(pos, cell):
    f = pos @ np.linalg.inv(cell)
    return (f - np.floor(f)) @ cell


def mic(d, cell):
    """Minimum-image vectors (fractional rounding; exact for |d| below the cell's inradius)."""
    f = d @ np.linalg.inv(cell)
    f = f - np.round(f)
    return f @ cell


def parabola_min(xs, es):
    """Vertex of the parabola through the lowest point and its two neighbours; edge -> that point."""
    i = int(np.argmin(es))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), True
    h = xs[1] - xs[0]
    den = es[i + 1] - 2 * es[i] + es[i - 1]
    if den <= 0:
        return float(xs[i]), True
    return float(xs[i] - 0.5 * h * (es[i + 1] - es[i - 1]) / den), False


# ----------------------------------------------------------------------------- metrics
def frame_metrics(r, r0, symbols, masses, cell):
    """Metrics of one frame of centroid positions r (unwrapped) against starting sites r0."""
    sym = np.asarray(symbols)
    m = np.asarray(masses, float)
    isH = sym == "H"
    disp = r - r0
    com = (m[:, None] * disp).sum(0) / m.sum()
    dn = np.linalg.norm(disp - com, axis=1)
    D = np.linalg.norm(mic(r[:, None, :] - r[None, :, :], cell), axis=2)
    np.fill_diagonal(D, np.inf)
    hh = D[np.ix_(isH, isH)]
    nn = hh.min(axis=1)
    bidx = np.where(sym != "H")[0]
    bmask = np.array([s not in ("H",) for s in sym])
    # B atoms = the transition metal (4a): the species that is neither H nor the 8c cation
    counts = {s: int(np.sum(sym == s)) for s in set(sym)}
    nH = int(isH.sum())
    Bsp = [s for s in counts if s != "H" and counts[s] * 6 == nH]
    if Bsp:
        bi = np.where(sym == Bsp[0])[0]
        nclose = (D[np.ix_(bi, np.where(isH)[0])] < CLUSTER_R).sum(axis=1)
        clusters = float(np.mean(nclose == 6))
    else:
        clusters = float("nan")
    return {
        "dimer": float(np.mean(nn < DIMER_R)),
        "wander": float(np.mean(dn[isH] > WANDER_R)),
        "host_out": float(np.mean(dn[~isH] > WANDER_R)),
        "h_disp": float(np.mean(dn[isH])),
        "host_disp": float(np.mean(dn[~isH])),
        "min_hh": float(nn.min()),
        "min_heavy": float(D[bidx].min()) if bmask.any() else float("inf"),
        "clusters": clusters,
    }


def classify(D, W, F, unphysical):
    """Run class (PREREG-v0.1 sec. 4) from window means D (dimer), W (wander), F (host_out)."""
    if unphysical:
        return "UNPHYSICAL"
    if F >= 0.25:
        return "HOST-COLLAPSE"
    if D >= 0.10 or W >= 0.20:
        return "H-BREAKDOWN"
    if D < 0.02 and W < 0.05 and F < 0.05:
        return "INTACT"
    return "PARTIAL"


def summarize(frames, times_fs, length_fs, min_len_fs=500.0, forces_bad=False):
    """Window = second half of the achieved run. Returns dict with class (or NOT GRADABLE)."""
    t = np.asarray(times_fs, float)
    out = {"achieved_fs": float(length_fs)}
    unphys = bool(forces_bad) or any(fr["min_hh"] < MIN_HH or fr["min_heavy"] < MIN_HEAVY
                                     for fr in frames)
    out["unphysical"] = unphys
    if length_fs < min_len_fs and not unphys:
        out["class"] = "NOT GRADABLE"
        return out
    sel = [fr for fr, ti in zip(frames, t) if ti >= 0.5 * length_fs]
    if not sel:
        sel = frames[-1:]
    for key in ("dimer", "wander", "host_out", "h_disp", "host_disp", "clusters"):
        out[key] = float(np.mean([fr[key] for fr in sel]))
    out["min_hh"] = float(min(fr["min_hh"] for fr in frames))
    out["min_heavy"] = float(min(fr["min_heavy"] for fr in frames))
    out["n_window_frames"] = len(sel)
    out["class"] = classify(out["dimer"], out["wander"], out["host_out"], unphys)
    return out


UNSTABLE = ("H-BREAKDOWN", "HOST-COLLAPSE")


def control_gate(md_au, pimd_au, pimd_ag):
    """PREREG-v0.1 sec. 5. Reproduce the published DFT-PIMD verdicts of arXiv:2604.12367."""
    g1 = pimd_ag in UNSTABLE
    g2 = md_au in ("INTACT", "PARTIAL") and pimd_au == "H-BREAKDOWN"
    if "NOT GRADABLE" in (md_au, pimd_au, pimd_ag):      # a run too short (budget): no verdict
        gate = "NOT GRADABLE"
    else:
        gate = "PASS" if (g1 and g2) else "FAIL"
    return {"G1_Li2AgH6_pimd_unstable": g1, "G2_Li2AuH6_classical_ok_quantum_breaks": g2,
            "gate": gate}


def target_verdict(gate, md_ir, pimd_ir):
    """Per-model verdict on Mg2IrH6 at 80 K (PREREG-v0.1 sec. 6)."""
    if gate != "PASS":
        return "NOT GRADABLE (control gate %s)" % gate
    if md_ir not in ("INTACT", "PARTIAL"):
        return "NOT GRADABLE (classical MD disagrees with published MD)"
    if pimd_ir == "INTACT":
        return "SURVIVES"
    if pimd_ir in UNSTABLE:
        return "KINETICALLY UNSTABLE"
    if pimd_ir == "PARTIAL":
        return "PARTIAL"
    return "NOT GRADABLE (%s)" % pimd_ir


def combine(verdicts):
    """Combine per-model verdicts (PREREG-v0.1 sec. 6)."""
    good = [v for v in verdicts if not v.startswith("NOT GRADABLE")]
    if not good:
        return "NOT GRADABLE"
    if len(set(good)) > 1:
        return "INCONCLUSIVE (models disagree)"
    return good[0] + (" (two models)" if len(good) >= 2 else " (one model)")
