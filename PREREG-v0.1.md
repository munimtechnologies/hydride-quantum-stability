# Does Mg2IrH6 survive quantum nuclear motion? Pre-registration v0.1

MV2 of Munim, Inc. (mv2@munimtech.com). 2026-11-22. Made with AI agents working in MV2's name;
**not yet checked by a materials physicist**. Text CC BY 4.0; code Apache-2.0.

Fixed before any simulation of this project has run. No result of this test exists while this is
written.

## 1. The question

Cubic Fm-3m **Mg2IrH6** is the best-known prediction of a hydride superconductor that works at
ambient pressure: Tc of about 160 K in Dolui et al., *Phys. Rev. Lett.* 132, 166001 (2024),
arXiv:2310.07562, and 65-160 K across later work (Sanna et al. 2024; arXiv:2310.06804 and
others). It is not thermodynamically stable. Its case rests on:

- dynamical stability, including a quantum anharmonic (SSCHA) calculation done **at 0 K**, which
  barely changes its phonons;
- **classical** molecular dynamics with a machine-learned potential (EDDP) at 100-1200 K, which
  showed "very little diffusion" at 300 K and an H-diffusion barrier of 0.56 eV.

In 2026, Ding, Chen and Shi (arXiv:2604.12367; *Phys. Rev. B*, doi 10.1103/w7rx-sknt) showed that
this kind of evidence can miss a real failure. They ran **path-integral MD** (PIMD: quantum
nuclei), with forces from direct DFT, on two related predicted ambient-pressure superconductors,
Li2AuH6 and Li2AgH6, at 80 K:

- Li2AgH6: the lattice collapses (classical MD and PIMD).
- Li2AuH6: intact in classical MD; under PIMD the Li-Au fluorite frame survives but hydrogen
  leaves its sites, 23.6 % of H atoms pair into H2 molecules, and the predicted Tc falls from
  80-140 K to 22 K.

Mg2IrH6, the flagship of the family, has not been given this test (searched 2026-11-22: arXiv,
journals, web; no PIMD or other path-integral study of Mg2IrH6, Mg2RhH6, Mg2PtH6 or MgAlFeH6
found). **The question: under quantum nuclear motion at ambient pressure, does Mg2IrH6 keep its
structure (as Li2AuH6 did not)?** Either answer matters to the leading ambient-pressure
superconductor candidate.

## 2. What we can and cannot do

We cannot run DFT-PIMD (16 beads x 72 atoms x thousands of steps) on the CPUs we have. We use
**universal machine-learned interatomic potentials** (MLIPs) trained on PBE data instead, and we
do not trust them blindly: MLIPs can be too soft or have "holes" for hydrogen-rich metastable
structures. So the two compounds of Ding et al. are run first as **positive controls**, under the
same settings as their DFT-PIMD. A model's verdict on Mg2IrH6 counts only if it reproduces the
published DFT verdicts on both controls.

Models (two independent families, both permissively licensed):

1. **MACE-MPA-0 medium** (MIT; MPtrj + sAlex), checkpoint
   `https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model`
   (sha256 printed by the job), float32.
2. **MatterSim v1.0.0 5M** (MIT; trained with finite-temperature and finite-pressure
   configurations), `MatterSim-v1.0.0-5M.pth` from the `mattersim` package (sha256 printed).

## 3. Simulations (code: `hpimd_core.py` 0.1.0, `hpimd_sim.py` 0.1.0)

- Structures: Fm-3m A2BH6, B at 4a, A at 8c, H at 24e (x,0,0); 2x2x2 supercell of the fcc
  primitive cell (72 atoms), as Ding et al. Fixed cell at the **published DFT (PBE) lattice
  constant**: Li2AgH6 a = 4.62 sqrt2 A, Li2AuH6 a = 4.71 sqrt2 A (Ding et al.), Mg2IrH6
  a = 6.66 A (Dolui et al.). NVT, like Ding et al. The H coordinate x is relaxed with each MLIP
  (scan 0.220-0.300 in steps of 0.0025, parabolic minimum). The MLIP's own zero-pressure lattice
  constant is printed for information only.
- PIMD: 16 beads, PILE-L thermostat (Ceriotti et al. 2010) with critically damped internal modes
  and centroid friction 10/ps; timestep 0.5 fs; all beads start on the ideal sites; seeds fixed
  (20261122 + run index). Classical MD = the same integrator with one bead.
- Runs: controls (job kind `ctrl`): MD and PIMD of Li2AuH6 and Li2AgH6 at 80 K. Target (kind
  `tgt`): MD and PIMD of Mg2IrH6 at 80 K (**primary**), PIMD at 300 K (**secondary, information
  only**).
- Length: MD 1.0 ps. PIMD 1.0 ps (Ding et al.'s preliminary runs were 1 ps on the same cell),
  shortened **uniformly** to the largest of 0.8 / 0.6 / 0.5 ps that the job's measured speed fits
  in 85 % of its budget (rule in `hpimd_sim.py`, decided from timing only, before any trajectory).
  The 300 K run is dropped first if it does not fit. A run that ends below 0.5 ps (budget) is
  **NOT GRADABLE** and is resumed by a new job version, never graded.

## 4. Metrics and run classes (thresholds fixed here)

All from ring-polymer **centroids**, averaged over the **second half** of each run, displacements
from the starting sites after removing the centre-of-mass drift, distances by minimum image:

- **D** dimer fraction: H atoms with another H closer than 1.0 A.
- **W** wander fraction: H atoms more than 1.0 A from their starting site.
- **F** host out-of-site fraction: non-H atoms more than 1.0 A from their starting site.
- Information only: intact BH6 clusters (6 H within 1.9 A of B, Dolui et al.'s definition),
  mean displacements, H ring-polymer radius of gyration, kinetic temperature.
- **UNPHYSICAL** (model failure) if any centroid H-H pair comes closer than 0.45 A, any pair
  involving a non-H atom closer than 0.90 A, or any force is non-finite or above 100 eV/A. The run
  stops there.

Class: UNPHYSICAL; else **HOST-COLLAPSE** if F >= 0.25; else **H-BREAKDOWN** if D >= 0.10 or
W >= 0.20; else **INTACT** if D < 0.02, W < 0.05 and F < 0.05; else **PARTIAL**.

## 5. Control gate (per model)

Published DFT-PIMD verdicts (Ding et al.) that the model must reproduce:

- **G1**: Li2AgH6 PIMD at 80 K is HOST-COLLAPSE or H-BREAKDOWN (kinetically unstable).
- **G2**: Li2AuH6 classical MD at 80 K is INTACT or PARTIAL, **and** Li2AuH6 PIMD at 80 K is
  H-BREAKDOWN (quantum motion breaks the H sublattice that classical motion keeps).

Gate PASS iff G1 and G2. If any of the three runs is NOT GRADABLE, the gate is NOT GRADABLE
(resume). Li2AgH6 classical MD is reported against the paper (collapse) but is not part of the
gate. Li2AuH6's PIMD dimer fraction is reported next to the published 23.6 % (information only;
our window and length differ).

## 6. Verdict on Mg2IrH6 (80 K, primary)

Per model: gate not PASS -> NOT GRADABLE. Classical MD of Mg2IrH6 at 80 K not INTACT/PARTIAL
(that would contradict the published classical MD) -> NOT GRADABLE. Otherwise the PIMD class
gives: INTACT -> **SURVIVES**; H-BREAKDOWN or HOST-COLLAPSE -> **KINETICALLY UNSTABLE**;
PARTIAL -> **PARTIAL**; UNPHYSICAL -> NOT GRADABLE.

Combined: gradable verdicts that agree -> that verdict ("two models" or "one model"); gradable
verdicts that disagree -> **INCONCLUSIVE**; none gradable -> NOT GRADABLE. If both gates FAIL, we
report that two universal MLIPs cannot reproduce the published DFT-PIMD verdicts on these
hydrides (a negative result about the method), and no claim about Mg2IrH6 is made.
`grade_hpimd.py` applies these rules to the jobs' `hpimd.json` files.

## 7. Predictions (ours, before data)

- Control gate PASS: MACE-MPA-0 35 %, MatterSim 45 %; at least one model fails to install or
  run on the first try: 30 %.
- Given at least one gated model, Mg2IrH6 at 80 K: SURVIVES 65 %, PARTIAL 20 %, KINETICALLY
  UNSTABLE 15 %. Reason: H in Mg2IrH6 sits in strongly bound [IrH6] units (Ir-H about 1.72 A,
  classical barrier 0.56 eV), unlike the weakly bound hypervalent [AuH6] unit.
- Secondary 300 K run: same class as 80 K, 75 %.

## 8. Limits, stated in advance

Universal MLIPs, not DFT; a fixed cell at the DFT lattice constant (as Ding et al.); 72 atoms
and about 1 ps, enough to see prompt breakdown, not slow diffusion; a PASS gate shows that a model
reproduces two published verdicts, not that it is right for Mg2IrH6. A SURVIVES verdict would not
mean that Mg2IrH6 can be made, and says nothing about its Tc. A KINETICALLY UNSTABLE verdict would
need DFT-PIMD confirmation before anyone should rely on it.

## 9. Jobs

`hpimd-ctrl-mpa-v1`, `hpimd-tgt-mpa-v1`, `hpimd-ctrl-msim-v1`, `hpimd-tgt-msim-v1` (4 CPUs, 4 GB,
budget 17,000 s each). Each first runs `py_compile` and the 13 unit tests of `test_hpimd_core.py`
(the PIMD integrator reproduces the exact quantum and classical harmonic-oscillator results), and
prints every number a grade needs on lines starting `HPIMD_`.
