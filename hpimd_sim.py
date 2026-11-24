# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""hpimd_sim 0.1.1 -- the pre-registered MLIP MD / PIMD runs of PREREG-v0.1 on CPU.

    python hpimd_sim.py --model mace-mpa-0   --kind ctrl --runs 0,1 --out /out --workers 2
    python hpimd_sim.py --model mattersim-5m --kind tgt  --runs 0,1 --out /out --workers 2

kind ctrl: runs 0 md-Li2AuH6, 1 md-Li2AgH6, 2 pimd-Li2AuH6, 3 pimd-Li2AgH6 (80 K, control gate);
kind tgt : runs 0 md-Mg2IrH6 80 K, 1 pimd-Mg2IrH6 80 K (primary), 2 pimd-Mg2IrH6 300 K
           (secondary, information only).

Changes from 0.1.0 (AMENDMENT-HPIMD-0.2.md), none of which touches the physics:
- a worker that dies (e.g. killed for memory) now stops the job with HPIMD_ERROR and exit 3,
  instead of hanging the pool until the job's timeout (what happened to hpimd-ctrl-mpa-v1);
- --runs selects a subset of the pre-registered runs so one job does what fits its budget;
  every run keeps its 0.1.0 seed (20261122 + its index in the full list);
- default 2 workers (the containers get 2 CPUs); worker peak memory is printed (HPIMD_MEM);
- if the measured speed cannot reach the 0.5 ps gradable minimum, the job stops before any
  trajectory with HPIMD_PLAN fits=false and exit 4, instead of running to NOT GRADABLE.
The integrator, structures, metrics and thresholds are hpimd_core 0.1.0, unchanged.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpimd_core as hc  # noqa: E402

VERSION = "0.1.1"
MACE_URL = {
    "mace-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
}
MODELS = ("mace-mpa-0", "mattersim-5m")
RUNS = {   # (mode, system, T, primary) -- identical to 0.1.0
    "ctrl": [("md", "Li2AuH6", 80.0, True), ("md", "Li2AgH6", 80.0, True),
             ("pimd", "Li2AuH6", 80.0, True), ("pimd", "Li2AgH6", 80.0, True)],
    "tgt": [("md", "Mg2IrH6", 80.0, True), ("pimd", "Mg2IrH6", 80.0, True),
            ("pimd", "Mg2IrH6", 300.0, False)],
}
SEED0 = 20261122
P_BEADS = 16
DT = 0.5            # fs
GAMMA0 = 0.01       # 1/fs (10 / ps, as arXiv:2604.12367)
STRIDE = 10         # steps between saved frames (5 fs)
MD_FS = 1000.0
PIMD_CHOICES = (1000.0, 800.0, 600.0, 500.0)
MIN_FS = 500.0

_CALC = None


class WorkerDied(RuntimeError):
    pass


def select_runs(kind, spec):
    """[(index, run)] for a comma list of indices into RUNS[kind]; 'all' or '' = every run.
    Indices keep their place in the full list (so seeds are those of 0.1.0)."""
    full = RUNS[kind]
    if spec in (None, "", "all"):
        return list(enumerate(full))
    idx = []
    for tok in str(spec).split(","):
        i = int(tok.strip())
        if i < 0 or i >= len(full):
            raise ValueError("run index %d out of range for kind %s" % (i, kind))
        if i in idx:
            raise ValueError("run index %d given twice" % i)
        idx.append(i)
    return [(i, full[i]) for i in sorted(idx)]


def plan_lengths(t_md, t_pimd, avail, runs):
    """0.1.0 length rule (largest PIMD length of PIMD_CHOICES that fits; secondary dropped
    first) plus 0.1.1's fits flag: can the primary runs reach MIN_FS at all?"""
    md_cost = sum(MD_FS / DT * t_md for r in runs if r[0] == "md")
    prim = [r for r in runs if r[0] == "pimd" and r[3]]
    sec = [r for r in runs if r[0] == "pimd" and not r[3]]
    L = PIMD_CHOICES[-1]
    for cand in PIMD_CHOICES:
        if md_cost + len(prim) * cand / DT * t_pimd <= avail:
            L = cand
            break
    include_sec = md_cost + (len(prim) + len(sec)) * L / DT * t_pimd <= avail
    fits = md_cost + len(prim) * MIN_FS / DT * t_pimd <= avail
    est = md_cost + (len(prim) + (len(sec) if include_sec else 0)) * L / DT * t_pimd
    return {"pimd_fs": L, "include_secondary": bool(include_sec), "fits": bool(fits),
            "md_cost_s": md_cost, "estimate_s": est}


def make_calc(model, model_path):
    if model == "mace-mpa-0":
        from mace.calculators import mace_mp
        return mace_mp(model=model_path, default_dtype="float32", device="cpu")
    if model == "mattersim-5m":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cpu")
    raise ValueError("unknown model " + model)


def _init(model, model_path):
    global _CALC
    import torch
    torch.set_num_threads(1)
    _CALC = make_calc(model, model_path)


def _eval(args):
    symbols, cell, pos = args
    from ase import Atoms
    at = Atoms(symbols=list(symbols), positions=hc.wrap(np.asarray(pos), cell), cell=cell,
               pbc=True)
    at.calc = _CALC
    e = float(at.get_potential_energy())
    f = np.asarray(at.get_forces(), float)
    return e, f


def _rss(_=None):
    import resource
    return os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MB


def pmap(ex, fn, items):
    try:
        return list(ex.map(fn, items, chunksize=1))
    except BrokenProcessPool as exc:
        raise WorkerDied(str(exc) or "a worker process died")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(model):
    if model not in MACE_URL:
        return None
    url = MACE_URL[model]
    path = os.path.join(tempfile.gettempdir(), "hpimd-models", os.path.basename(url))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k in range(6):
        try:
            urllib.request.urlretrieve(url, path)
            break
        except Exception as exc:  # pragma: no cover - network
            print("model download attempt %d failed: %s" % (k + 1, exc), flush=True)
            time.sleep(30)
    else:
        raise SystemExit("could not download " + url)
    print("HPIMD_MODEL %s url=%s bytes=%d sha256=%s" % (
        model, url, os.path.getsize(path), sha256_file(path)), flush=True)
    return path


def versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for pkg in ("torch", "ase", "mace-torch", "e3nn", "mattersim", "torch_geometric"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            pass
    return out


def prep_system(ex, name):
    """MLIP-relaxed H position x (24e) at the published DFT lattice, and the MLIP's own
    zero-pressure lattice constant (information only). Same as 0.1.0."""
    s = hc.SYSTEMS[name]
    xs = np.round(np.arange(0.220, 0.30001, 0.0025), 5)
    jobs = []
    for x in xs:
        sym, pos, cell = hc.a2bh6(s["A"], s["B"], s["a"], x, n=1)
        jobs.append((tuple(sym), cell, pos))
    ex_ = np.array([r[0] for r in pmap(ex, _eval, jobs)])
    xb, xedge = hc.parabola_min(xs, ex_)
    scales = np.round(np.arange(0.96, 1.04001, 0.01), 4)
    jobs = []
    for sc in scales:
        sym, pos, cell = hc.a2bh6(s["A"], s["B"], s["a"] * sc, xb, n=1)
        jobs.append((tuple(sym), cell, pos))
    ea = np.array([r[0] for r in pmap(ex, _eval, jobs)])
    sb, aedge = hc.parabola_min(scales, ea)
    out = {"a_dft": float(s["a"]), "a_source": s["source"], "x_best": xb, "x_edge": xedge,
           "B_H": xb * s["a"], "a_mlip": float(s["a"] * sb), "a_edge": aedge,
           "x_scan": [[float(a), float(b)] for a, b in zip(xs, ex_)],
           "a_scan": [[float(a * s["a"]), float(b)] for a, b in zip(scales, ea)]}
    print("HPIMD_PREP %s a_dft=%.4f x_best=%.5f edge=%s B-H=%.4f a_mlip=%.4f (%.2f%% vs DFT) "
          "a_edge=%s" % (name, s["a"], xb, xedge, xb * s["a"], out["a_mlip"],
                         100 * (sb - 1), aedge), flush=True)
    return out


def run_one(ex, mode, name, T, length_fs, x_best, seed, deadline, outdir):
    s = hc.SYSTEMS[name]
    sym, r0, cell = hc.a2bh6(s["A"], s["B"], s["a"], x_best, n=2)
    symt = tuple(sym)
    m = np.array([hc.MASS[x] for x in sym])
    isH = np.array([x == "H" for x in sym])
    P = 1 if mode == "md" else P_BEADS
    tag = "%s-%s-%dK" % (mode, name, int(T))

    def ff(q):
        res = pmap(ex, _eval, [(symt, cell, q[j]) for j in range(q.shape[0])])
        return np.stack([r[1] for r in res]), np.array([r[0] for r in res])

    t_start = time.time()
    rp = hc.RingPolymerLangevin(r0, m, P, T, DT, GAMMA0, ff, seed)
    nsteps = int(round(length_fs / DT))
    frames, times, cents, extra = [], [], [], []
    bad = not rp.forces_ok()
    stop = "forces" if bad else ""

    def record():
        c = rp.centroid()
        fm = hc.frame_metrics(c, r0, sym, m, cell)
        rg = float(np.sqrt(np.mean(np.sum((rp.q[:, isH] - c[None, isH]) ** 2, axis=2))))
        frames.append(fm)
        times.append(rp.nstep * DT)
        cents.append(c.astype(np.float32))
        extra.append((float(np.mean(rp.e)) / len(sym), rp.kinetic_T(), rg))
        return fm

    record()
    while not bad and rp.nstep < nsteps:
        if time.time() > deadline:
            stop = "budget"
            break
        rp.step()
        if not rp.forces_ok():
            bad, stop = True, "forces"
            break
        if rp.nstep % STRIDE == 0:
            fm = record()
            if fm["min_hh"] < hc.MIN_HH or fm["min_heavy"] < hc.MIN_HEAVY:
                stop = "unphysical contact"
                break
        if rp.nstep % 200 == 0:
            fm = frames[-1]
            print("  %s t=%.0f fs  T_kin=%.1f K  epot/atom=%.4f eV  D=%.3f W=%.3f F=%.3f "
                  "clusters=%.3f min_hh=%.3f h_rg=%.3f  %.0f s" % (
                      tag, rp.nstep * DT, extra[-1][1], extra[-1][0], fm["dimer"], fm["wander"],
                      fm["host_out"], fm["clusters"], fm["min_hh"], extra[-1][2],
                      time.time() - t_start), flush=True)
    achieved = rp.nstep * DT
    summ = hc.summarize(frames, times, achieved, min_len_fs=MIN_FS, forces_bad=bad)
    ex_ = np.array(extra)
    half = np.array(times) >= 0.5 * achieved
    summ.update({"tag": tag, "mode": mode, "system": name, "T": T, "P": P,
                 "planned_fs": length_fs, "stop": stop or "complete", "seed": seed,
                 "seconds": time.time() - t_start,
                 "T_kin_window": float(ex_[half, 1].mean()) if half.any() else float("nan"),
                 "h_rg_window": float(ex_[half, 2].mean()) if half.any() else float("nan"),
                 "epot_first": float(ex_[0, 0]), "epot_last": float(ex_[-1, 0])})
    np.savez_compressed(os.path.join(outdir, tag + ".npz"), centroids=np.array(cents),
                        times=np.array(times), r0=r0, cell=cell, symbols=np.array(sym))
    with open(os.path.join(outdir, tag + "_frames.csv"), "w") as fh:
        fh.write("t_fs,dimer,wander,host_out,h_disp,host_disp,min_hh,min_heavy,clusters,"
                 "epot_per_atom,T_kin,h_rg\n")
        for t, fm, e in zip(times, frames, extra):
            fh.write("%.1f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.6f,%.2f,%.5f\n" % (
                t, fm["dimer"], fm["wander"], fm["host_out"], fm["h_disp"], fm["host_disp"],
                fm["min_hh"], fm["min_heavy"], fm["clusters"], e[0], e[1], e[2]))
    print("HPIMD_RUN %s P=%d planned_fs=%.0f achieved_fs=%.1f stop=%s class=%s D=%s W=%s F=%s "
          "clusters=%s min_hh=%.3f min_heavy=%.3f T_kin=%.1f h_rg=%.3f secs=%.0f" % (
              tag, P, length_fs, achieved, summ["stop"], summ["class"],
              _f(summ.get("dimer")), _f(summ.get("wander")), _f(summ.get("host_out")),
              _f(summ.get("clusters")), min(fr["min_hh"] for fr in frames),
              min(fr["min_heavy"] for fr in frames), summ["T_kin_window"], summ["h_rg_window"],
              summ["seconds"]), flush=True)
    return summ


def _f(v):
    return "NA" if v is None else "%.4f" % v


def write_json(result, outdir):
    with open(os.path.join(outdir, "hpimd.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)


def simulate(args, result, t0, deadline, model_path, ctx):
    runs = select_runs(args.kind, args.runs)
    result["selected"] = [i for i, _ in runs]
    print("HPIMD_SELECTED " + json.dumps([[i, "%s-%s-%dK" % (r[0], r[1], int(r[2]))]
                                          for i, r in runs]), flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init,
                             initargs=(args.model, model_path)) as ex:
        prep = {}
        for name in sorted({r[1] for _, r in runs}):
            prep[name] = prep_system(ex, name)
        result["prep"] = prep
        name0 = runs[0][1][1]
        s = hc.SYSTEMS[name0]
        sym, pos, cell = hc.a2bh6(s["A"], s["B"], s["a"], prep[name0]["x_best"], n=2)
        item = (tuple(sym), cell, pos)
        pmap(ex, _eval, [item] * args.workers)
        t_md = t_pimd = 0.0
        if any(r[0] == "md" for _, r in runs):
            tb = time.time()
            for _ in range(5):
                pmap(ex, _eval, [item])
            t_md = (time.time() - tb) / 5
        if any(r[0] == "pimd" for _, r in runs):
            tb = time.time()
            for _ in range(3):
                pmap(ex, _eval, [item] * P_BEADS)
            t_pimd = (time.time() - tb) / 3
        mem = {}
        for pid, mb in pmap(ex, _rss, list(range(4 * args.workers))):
            mem[pid] = max(mem.get(pid, 0.0), mb)
        result["worker_peak_mb"] = sorted(mem.values())
        print("HPIMD_MEM workers_seen=%d worker_peak_mb=%s main_peak_mb=%.0f" % (
            len(mem), json.dumps([round(v) for v in sorted(mem.values())]), _rss()[1]),
            flush=True)
        used = time.time() - t0
        avail = 0.85 * args.budget_s - used
        plan = plan_lengths(t_md, t_pimd, avail, [r for _, r in runs])
        plan.update({"t_md_call_s": t_md, "t_pimd_step_s": t_pimd, "setup_s": used,
                     "avail_s": avail})
        result["plan"] = plan
        print("HPIMD_PLAN " + json.dumps(plan, sort_keys=True), flush=True)
        if not plan["fits"]:
            print("HPIMD_STOP the primary runs cannot reach %.0f fs in this budget; no "
                  "trajectory was started (exit 4)" % MIN_FS, flush=True)
            return 4
        for i, (mode, name, T, primary) in runs:
            tag = "%s-%s-%dK" % (mode, name, int(T))
            if not primary and not plan["include_secondary"]:
                print("HPIMD_RUN %s skipped (secondary run does not fit the budget)" % tag,
                      flush=True)
                continue
            length = MD_FS if mode == "md" else plan["pimd_fs"]
            summ = run_one(ex, mode, name, T, length, prep[name]["x_best"], SEED0 + i,
                           deadline, args.out)
            summ["primary"] = primary
            summ["run_index"] = i
            result["runs"].append(summ)
            write_json(result, args.out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--kind", required=True, choices=sorted(RUNS))
    ap.add_argument("--runs", default="all", help="comma list of run indices, or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=17000.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    select_runs(args.kind, args.runs)          # reject a bad --runs before any download
    t0 = time.time()
    deadline = t0 + args.budget_s
    os.makedirs(args.out, exist_ok=True)
    print("hpimd_sim %s, hpimd_core %s, model %s, kind %s, runs %s, workers %d, budget %.0f s" % (
        VERSION, hc.VERSION, args.model, args.kind, args.runs, args.workers, args.budget_s),
        flush=True)
    model_path = fetch_model(args.model)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx, initializer=_init,
                             initargs=(args.model, model_path)) as warm:   # downloads happen once
        pmap(warm, _eval, [(("H", "H"), 10.0 * np.eye(3), np.array([[0, 0, 0], [0.74, 0, 0]]))])
    vers = versions()
    print("HPIMD_VERSIONS " + json.dumps(vers, sort_keys=True), flush=True)
    for p in glob.glob(os.path.expanduser("~/.local/mattersim/pretrained_models/*")):
        print("HPIMD_MODEL %s file=%s bytes=%d sha256=%s" % (
            args.model, os.path.basename(p), os.path.getsize(p), sha256_file(p)), flush=True)
    result = {"hpimd_sim": VERSION, "hpimd_core": hc.VERSION, "model": args.model,
              "kind": args.kind, "versions": vers, "runs": []}
    try:
        rc = simulate(args, result, t0, deadline, model_path, ctx)
    except WorkerDied as exc:
        result["error"] = "worker died: %s" % exc
        print("HPIMD_ERROR a worker process died (likely killed for memory): %s" % exc,
              flush=True)
        rc = 3
    classes = {r["tag"]: r["class"] for r in result["runs"]}
    print("HPIMD_CLASSES model=%s kind=%s %s" % (args.model, args.kind,
                                                 json.dumps(classes, sort_keys=True)), flush=True)
    result["total_s"] = time.time() - t0
    result["exit"] = rc
    write_json(result, args.out)
    print("HPIMD_DONE total_s=%.0f exit=%d" % (result["total_s"], rc), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
