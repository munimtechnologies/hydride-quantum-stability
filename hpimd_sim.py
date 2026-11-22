# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""hpimd_sim 0.1.0 -- the pre-registered MLIP MD / PIMD runs of PREREG-v0.1 on CPU.

    python hpimd_sim.py --model mace-mpa-0   --kind ctrl --out /out --budget-s 17000 --workers 4
    python hpimd_sim.py --model mattersim-5m --kind tgt  --out /out --budget-s 17000 --workers 4

kind ctrl: Li2AuH6 and Li2AgH6, classical MD and 16-bead PIMD at 80 K (control gate);
kind tgt : Mg2IrH6, classical MD and PIMD at 80 K (primary) and PIMD at 300 K (secondary).
Forces come from a universal machine-learned potential, evaluated bead by bead in a pool of
single-threaded worker processes. Every number a grade needs is printed (lines HPIMD_*).
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpimd_core as hc  # noqa: E402

VERSION = "0.1.0"
MACE_URL = {
    "mace-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
}
MODELS = ("mace-mpa-0", "mattersim-5m")
RUNS = {   # (mode, system, T, primary)
    "ctrl": [("md", "Li2AuH6", 80.0, True), ("md", "Li2AgH6", 80.0, True),
             ("pimd", "Li2AuH6", 80.0, True), ("pimd", "Li2AgH6", 80.0, True)],
    "tgt": [("md", "Mg2IrH6", 80.0, True), ("pimd", "Mg2IrH6", 80.0, True),
            ("pimd", "Mg2IrH6", 300.0, False)],
}
P_BEADS = 16
DT = 0.5            # fs
GAMMA0 = 0.01       # 1/fs (10 / ps, as arXiv:2604.12367)
STRIDE = 10         # steps between saved frames (5 fs)
MD_FS = 1000.0
PIMD_CHOICES = (1000.0, 800.0, 600.0, 500.0)
MIN_FS = 500.0

_CALC = None


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


def prep_system(pool, name):
    """MLIP-relaxed H position x (24e) at the published DFT lattice, and the MLIP's own
    zero-pressure lattice constant (information only)."""
    s = hc.SYSTEMS[name]
    xs = np.round(np.arange(0.220, 0.30001, 0.0025), 5)
    jobs = []
    for x in xs:
        sym, pos, cell = hc.a2bh6(s["A"], s["B"], s["a"], x, n=1)
        jobs.append((tuple(sym), cell, pos))
    ex = np.array([r[0] for r in pool.map(_eval, jobs, chunksize=1)])
    xb, xedge = hc.parabola_min(xs, ex)
    scales = np.round(np.arange(0.96, 1.04001, 0.01), 4)
    jobs = []
    for sc in scales:
        sym, pos, cell = hc.a2bh6(s["A"], s["B"], s["a"] * sc, xb, n=1)
        jobs.append((tuple(sym), cell, pos))
    ea = np.array([r[0] for r in pool.map(_eval, jobs, chunksize=1)])
    sb, aedge = hc.parabola_min(scales, ea)
    out = {"a_dft": float(s["a"]), "a_source": s["source"], "x_best": xb, "x_edge": xedge,
           "B_H": xb * s["a"], "a_mlip": float(s["a"] * sb), "a_edge": aedge,
           "x_scan": [[float(a), float(b)] for a, b in zip(xs, ex)],
           "a_scan": [[float(a * s["a"]), float(b)] for a, b in zip(scales, ea)]}
    print("HPIMD_PREP %s a_dft=%.4f x_best=%.5f edge=%s B-H=%.4f a_mlip=%.4f (%.2f%% vs DFT) "
          "a_edge=%s" % (name, s["a"], xb, xedge, xb * s["a"], out["a_mlip"],
                         100 * (sb - 1), aedge), flush=True)
    return out


def run_one(pool, mode, name, T, length_fs, x_best, seed, deadline, outdir):
    s = hc.SYSTEMS[name]
    sym, r0, cell = hc.a2bh6(s["A"], s["B"], s["a"], x_best, n=2)
    symt = tuple(sym)
    m = np.array([hc.MASS[x] for x in sym])
    isH = np.array([x == "H" for x in sym])
    P = 1 if mode == "md" else P_BEADS
    tag = "%s-%s-%dK" % (mode, name, int(T))

    def ff(q):
        res = pool.map(_eval, [(symt, cell, q[j]) for j in range(q.shape[0])], chunksize=1)
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
    ex = np.array(extra)
    half = np.array(times) >= 0.5 * achieved
    summ.update({"tag": tag, "mode": mode, "system": name, "T": T, "P": P,
                 "planned_fs": length_fs, "stop": stop or "complete", "seed": seed,
                 "seconds": time.time() - t_start,
                 "T_kin_window": float(ex[half, 1].mean()) if half.any() else float("nan"),
                 "h_rg_window": float(ex[half, 2].mean()) if half.any() else float("nan"),
                 "epot_first": float(ex[0, 0]), "epot_last": float(ex[-1, 0])})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--kind", required=True, choices=sorted(RUNS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=17000.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + args.budget_s
    os.makedirs(args.out, exist_ok=True)
    print("hpimd_sim %s, hpimd_core %s, model %s, kind %s, workers %d, budget %.0f s" % (
        VERSION, hc.VERSION, args.model, args.kind, args.workers, args.budget_s), flush=True)
    model_path = fetch_model(args.model)
    ctx = mp.get_context("spawn")
    with ctx.Pool(1, _init, (args.model, model_path)) as warm:     # downloads, if any, happen once
        warm.map(_eval, [(("H", "H"), 10.0 * np.eye(3), np.array([[0, 0, 0], [0.74, 0, 0]]))])
    vers = versions()
    print("HPIMD_VERSIONS " + json.dumps(vers, sort_keys=True), flush=True)
    for p in glob.glob(os.path.expanduser("~/.local/mattersim/pretrained_models/*")):
        print("HPIMD_MODEL %s file=%s bytes=%d sha256=%s" % (
            args.model, os.path.basename(p), os.path.getsize(p), sha256_file(p)), flush=True)
    runs = RUNS[args.kind]
    result = {"hpimd_sim": VERSION, "hpimd_core": hc.VERSION, "model": args.model,
              "kind": args.kind, "versions": vers, "runs": []}
    with ctx.Pool(args.workers, _init, (args.model, model_path)) as pool:
        prep = {}
        for name in sorted({r[1] for r in runs}):
            prep[name] = prep_system(pool, name)
        result["prep"] = prep
        # benchmark on the first system's 72-atom cell
        name0 = runs[0][1]
        s = hc.SYSTEMS[name0]
        sym, pos, cell = hc.a2bh6(s["A"], s["B"], s["a"], prep[name0]["x_best"], n=2)
        item = (tuple(sym), cell, pos)
        pool.map(_eval, [item] * args.workers)
        tb = time.time()
        for _ in range(5):
            pool.map(_eval, [item])
        t_md = (time.time() - tb) / 5
        tb = time.time()
        for _ in range(3):
            pool.map(_eval, [item] * P_BEADS, chunksize=1)
        t_pimd = (time.time() - tb) / 3
        used = time.time() - t0
        avail = 0.85 * args.budget_s - used
        md_cost = sum(MD_FS / DT * t_md for r in runs if r[0] == "md")
        prim = [r for r in runs if r[0] == "pimd" and r[3]]
        sec = [r for r in runs if r[0] == "pimd" and not r[3]]
        L = PIMD_CHOICES[-1]
        for cand in PIMD_CHOICES:
            if md_cost + len(prim) * cand / DT * t_pimd <= avail:
                L = cand
                break
        include_sec = md_cost + (len(prim) + len(sec)) * L / DT * t_pimd <= avail
        plan = {"t_md_call_s": t_md, "t_pimd_step_s": t_pimd, "setup_s": used, "avail_s": avail,
                "pimd_fs": L, "include_secondary": include_sec,
                "estimate_s": md_cost + (len(prim) + (len(sec) if include_sec else 0)) * L / DT * t_pimd}
        result["plan"] = plan
        print("HPIMD_PLAN " + json.dumps(plan, sort_keys=True), flush=True)
        classes = {}
        for i, (mode, name, T, primary) in enumerate(runs):
            if not primary and not include_sec:
                print("HPIMD_RUN %s-%s-%dK skipped (secondary run does not fit the budget)" % (
                    mode, name, int(T)), flush=True)
                continue
            length = MD_FS if mode == "md" else L
            summ = run_one(pool, mode, name, T, length, prep[name]["x_best"], 20261122 + i,
                           deadline, args.out)
            summ["primary"] = primary
            result["runs"].append(summ)
            classes[summ["tag"]] = summ["class"]
            with open(os.path.join(args.out, "hpimd.json"), "w") as fh:
                json.dump(result, fh, indent=1, default=float)
    if args.kind == "ctrl":
        g = hc.control_gate(classes.get("md-Li2AuH6-80K", "NOT GRADABLE"),
                            classes.get("pimd-Li2AuH6-80K", "NOT GRADABLE"),
                            classes.get("pimd-Li2AgH6-80K", "NOT GRADABLE"))
        result["gate"] = g
        print("HPIMD_GATE model=%s gate=%s G1=%s G2=%s md_Au=%s pimd_Au=%s pimd_Ag=%s md_Ag=%s" % (
            args.model, g["gate"], g["G1_Li2AgH6_pimd_unstable"],
            g["G2_Li2AuH6_classical_ok_quantum_breaks"], classes.get("md-Li2AuH6-80K"),
            classes.get("pimd-Li2AuH6-80K"), classes.get("pimd-Li2AgH6-80K"),
            classes.get("md-Li2AgH6-80K")), flush=True)
    else:
        print("HPIMD_TARGET model=%s md80=%s pimd80=%s pimd300=%s" % (
            args.model, classes.get("md-Mg2IrH6-80K"), classes.get("pimd-Mg2IrH6-80K"),
            classes.get("pimd-Mg2IrH6-300K", "skipped")), flush=True)
    result["total_s"] = time.time() - t0
    with open(os.path.join(args.out, "hpimd.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print("HPIMD_DONE total_s=%.0f" % result["total_s"], flush=True)


if __name__ == "__main__":
    main()
