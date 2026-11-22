# Prior-art note, 2026-11-22 (before any hpimd job has returned)

Written while all four pre-registered jobs of PREREG-v0.1 (hpimd-ctrl-mpa-v1, hpimd-tgt-mpa-v1,
hpimd-ctrl-msim-v1, hpimd-tgt-msim-v1) are still queued; no number from them has been seen.
Nothing in PREREG-v0.1 changes. This note only records what the article must credit.

1. **Dolui et al. already checked quantum anharmonic effects with SSCHA.** In arXiv:2310.07562
   (PRL 132, 166001, 2024) the authors applied the stochastic self-consistent harmonic
   approximation to cubic Mg2IrH6 and report that quantum anharmonic effects do not
   significantly change the phonon dispersion and do not remove its dynamical stability at
   ambient pressure (internal pressure rises from 0 to about 2 GPa). Our pre-registered
   prediction (SURVIVES 65 %) is in line with this.
2. **What our test adds.** SSCHA treats quantum fluctuations around a single
   self-consistent harmonic reference. Our test runs finite-temperature path-integral MD
   (16 beads, 80 K and 300 K) with two universal machine-learned potentials, after a control
   gate that must reproduce the DFT-PIMD collapse of Li2AgH6 and the quantum-only breakdown of
   Li2AuH6 (Ding, Chen, Shi, arXiv:2604.12367). Large-amplitude hydrogen motion of the kind
   that broke Li2AuH6 is outside the SSCHA's harmonic reference and is what PIMD can see.
3. **Other related work found in the scoop check (2026-11-22):** arXiv:2604.25626 (harmonic
   phonons and Grüneisen-type anharmonicity of Mg2TmH6, Tm = Rh, Pd, Ir, Pt; Mg2IrH6 lowest
   γ = 1.70) and arXiv:2411.15683 (X2MH6 family screening). No path-integral MD of Mg2IrH6 was
   found.

The code review on 2026-11-22 found no defect in hpimd_core 0.1.0 or hpimd_sim 0.1.0
(ring-polymer normal modes, PILE-L thermostat constants, unit conversions, bead kinetic
temperature, Fm-3m structure builder and metrics were re-derived by hand).

Made with AI agents working in MV2's name; no expert has checked this yet.
