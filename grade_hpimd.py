# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""grade_hpimd 0.1.1 -- grade PREREG-v0.1 from the hpimd.json files of each model's jobs.

    python grade_hpimd.py --pair mace-mpa-0 ctrl_md.json,pau.json,pag.json tgt.json \
                          --pair mattersim-5m ctrl_md.json,pau.json,pag.json tgt.json

0.1.1 (AMENDMENT-HPIMD-0.2): the control runs of one model may come from several jobs, so CTRL
and TGT are comma lists of hpimd.json files whose runs are merged by tag. A missing file only
removes its runs (each missing run is NOT GRADABLE, never a pass or a fail). The same tag in two
files is a FATAL error. The gate and verdict rules are hpimd_core 0.1.0, unchanged.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpimd_core as hc  # noqa: E402

VERSION = "0.1.1"


def classes(paths):
    """Merged {tag: class} over a comma list of hpimd.json files; None if none exists."""
    if not paths:
        return None
    merged, seen = {}, False
    for p in [x for x in str(paths).split(",") if x]:
        if not os.path.exists(p):
            continue
        seen = True
        with open(p) as fh:
            res = json.load(fh)
        for r in res.get("runs", []):
            if r["tag"] in merged:
                raise SystemExit("FATAL run %s appears in more than one file" % r["tag"])
            merged[r["tag"]] = r["class"]
    return merged if seen else None


def grade_model(ctrl_paths, tgt_paths):
    c, t = classes(ctrl_paths), classes(tgt_paths)
    if c is None:
        gate = {"gate": "NOT GRADABLE"}
    else:
        gate = hc.control_gate(c.get("md-Li2AuH6-80K", "NOT GRADABLE"),
                               c.get("pimd-Li2AuH6-80K", "NOT GRADABLE"),
                               c.get("pimd-Li2AgH6-80K", "NOT GRADABLE"))
    if t is None:
        verdict = "NOT GRADABLE (no target job)"
    else:
        verdict = hc.target_verdict(gate["gate"], t.get("md-Mg2IrH6-80K", "NOT GRADABLE"),
                                    t.get("pimd-Mg2IrH6-80K", "NOT GRADABLE"))
    return gate, verdict, c, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", metavar=("MODEL", "CTRLS", "TGTS"),
                    required=True)
    args = ap.parse_args()
    verdicts = []
    for model, cp, tp in args.pair:
        gate, verdict, c, t = grade_model(cp, tp)
        verdicts.append(verdict)
        print("GRADE_MODEL %s gate=%s verdict=%s ctrl=%s tgt=%s" % (
            model, gate["gate"], verdict, json.dumps(c, sort_keys=True),
            json.dumps(t, sort_keys=True)))
    print("GRADE_COMBINED %s" % hc.combine(verdicts))


if __name__ == "__main__":
    main()
