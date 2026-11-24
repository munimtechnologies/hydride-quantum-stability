# Amendment HPIMD-0.2 to PREREG-v0.1 (Mg2IrH6 under quantum nuclear motion)

MV2 of Munim, Inc. (mv2@munimtech.com). 2026-11-24. Made with AI agents working in MV2's name;
**not yet checked by a materials physicist**. Text CC BY 4.0; code Apache-2.0.

Written after the first job failed and **before any MD or PIMD trajectory of this project exists**.
No run class, displacement or dimer fraction has been seen by anyone. Nothing in sections 1, 2, 4,
5, 6, 7 or 8 of PREREG-v0.1 changes: same structures, models, integrator, beads, timestep,
thermostat, seeds, metrics, thresholds, gate, verdict rules and predictions.

## What happened

`hpimd-ctrl-mpa-v1` (MACE-MPA-0, controls) started 2026-11-24 07:18 UTC on a 2-CPU, 4 GB
container (it asked for 4 CPUs) and was stopped at its 320-minute limit (exit 124; the runner's
reason: out of memory at the 4 GB cap). Its log shows:

1. the 13 unit tests passed; the model file was the pinned one
   (sha256 75428afe...4fb638);
2. the H-position scans finished for both controls (information-only numbers, seen now and
   disclosed here: the MLIP's own zero-pressure lattice constants are 0.49 % (Li2AgH6) and
   0.44 % (Li2AuH6) above the published DFT values; the runs use the DFT values, as registered);
3. then one of the four worker processes was replaced by a fresh one (the log shows a fifth model
   load) and nothing more was printed for about five hours.

Diagnosis: four single-threaded MACE workers, each holding torch, the model and the 72-atom
force graph, did not fit in 4 GB; one was killed, and Python's `multiprocessing.Pool` never
returns from `map` when a worker dies, so the job hung until its time limit. The budget check
in 0.1.0 sits only inside the MD loop, so it could not stop the hang. The three other queued
jobs (`hpimd-tgt-mpa-v1`, `hpimd-ctrl-msim-v1`, `hpimd-tgt-msim-v1`) use the same code and
settings and would fail the same way.

A second, separate problem: the containers give 2 CPUs, not 4. At 2 workers, the four control
runs of one model (two 1 ps MD runs and two 16-bead PIMD runs of 72 atoms) are unlikely to fit
in one job's budget at all.

## Changes (hpimd_sim 0.1.1, grade_hpimd 0.1.1)

1. **Crash-safe workers.** `concurrent.futures.ProcessPoolExecutor` replaces `multiprocessing.Pool`.
   A worker that dies raises at once; the job writes what it has, prints `HPIMD_ERROR` and exits 3.
2. **Two workers** (the CPUs actually given); peak worker memory is printed (`HPIMD_MEM`).
3. **One model's runs are split over jobs** with `--runs` (indices into the unchanged run list):
   | job | model | runs |
   |---|---|---|
   | `hpimd-ctrl-mpa-v2`, `hpimd-ctrl-msim-v2` | MACE / MatterSim | ctrl 0, 1: classical MD of Li2AuH6 and Li2AgH6 |
   | `hpimd-pau-mpa-v1`, `hpimd-pau-msim-v1` | MACE / MatterSim | ctrl 2: PIMD of Li2AuH6 |
   | `hpimd-pag-mpa-v1`, `hpimd-pag-msim-v1` | MACE / MatterSim | ctrl 3: PIMD of Li2AgH6 |
   | `hpimd-tgt-mpa-v2`, `hpimd-tgt-msim-v2` | MACE / MatterSim | tgt 0, 1: MD and PIMD of Mg2IrH6, 80 K |
   Each run keeps its 0.1.0 seed (20261122 + its index in the full list). The `-v2` names
   supersede the queued `-v1` jobs of the same name.
4. **PIMD length is chosen per job**, by the unchanged rule (largest of 1.0 / 0.8 / 0.6 / 0.5 ps
   that the job's measured speed fits in 85 % of its budget), so the two control PIMD runs of one
   model may end up with different lengths; each is graded on the second half of its own run,
   as registered. Any length difference will be reported next to the grades.
5. **Fail fast.** If the measured speed cannot reach the 0.5 ps gradable minimum, the job stops
   before any trajectory (`HPIMD_PLAN ... "fits": false`, exit 4). The next step is then a
   further split or a longer budget, decided from timing only.
6. **The 300 K secondary run (information only) is deferred.** It is not filed now; it may be
   filed later as its own job (tgt run 2) and remains information only.
7. **Grader.** `grade_hpimd.py` merges each model's runs from several `hpimd.json` files by tag
   (the same tag twice is FATAL; a missing file leaves its runs NOT GRADABLE).
8. **Frozen files are not retyped.** Each job downloads `hpimd_core.py`, `test_hpimd_core.py` and
   `PREREG-v0.1.md` from commit 8f76b8240a4d8807cae6102e0ca46dd2d4bcf4b7 of this repository and
   checks their git blob sha1 (`pin_hpimd.py`, FATAL on mismatch).

Tests: the 13 frozen tests of `test_hpimd_core.py` plus 13 new ones in `test_hpimd_split.py`
(run selection, the length rule including the new fits flag, the grader's merge) run first in
every job.

## Our updated expectations (not a change of predictions)

The PREREG-v0.1 predictions stand. We add one about the machinery, before any of the new jobs
runs: at least one of the eight jobs stops at the speed check (exit 4) or for memory, 45 %.
