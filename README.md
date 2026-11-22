# hydride-quantum-stability

**Does Mg2IrH6, the leading prediction of a hydride superconductor at ambient pressure, survive
quantum nuclear motion?** An open, pre-registered test by MV2 of Munim, Inc.

Mg2IrH6 was predicted to superconduct near 160 K without high pressure (Dolui et al., *PRL* 132,
166001, 2024). Its stability case rests on 0 K quantum-anharmonic phonons and on classical
molecular dynamics. In 2026, Ding, Chen and Shi (arXiv:2604.12367, *PRB*) showed with
path-integral MD (quantum nuclei) that two related predicted compounds, Li2AuH6 and Li2AgH6, are
kinetically unstable even though they had passed the same kinds of checks. In Li2AuH6, hydrogen
pairs into H2 molecules and the predicted Tc falls from 80-140 K to 22 K. We found no
path-integral study of Mg2IrH6.

This repository runs that test on CPUs with two universal machine-learned potentials
(MACE-MPA-0 and MatterSim v1 5M). Each model must first reproduce the published DFT
path-integral verdicts on Li2AuH6 and Li2AgH6 (a control gate) before its verdict on Mg2IrH6
counts. Everything is fixed in [PREREG-v0.1.md](PREREG-v0.1.md) before any run.

| File | What it is |
|---|---|
| `hpimd_core.py` | numpy-only ring-polymer MD (PILE-L), structures, metrics, classes, gate, verdicts |
| `test_hpimd_core.py` | 13 unit tests (exact quantum and classical harmonic oscillator, metrics, rules) |
| `hpimd_sim.py` | runs the MLIP MD / PIMD jobs on CPU and prints every graded number (`HPIMD_*`) |
| `grade_hpimd.py` | applies the pre-registered gate and verdict rules to the jobs' outputs |
| `jobs/*.json` | the exact compute jobs |

Status (2026-11-22): pre-registered; jobs queued. Results will be committed under `results/`
whichever way they come out.

Made with AI agents working in MV2's name; not yet checked by a materials physicist.
Code Apache-2.0; text CC BY 4.0. Contact: MV2, Munim, Inc., mv2@munimtech.com.
