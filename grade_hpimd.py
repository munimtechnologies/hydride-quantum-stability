# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""Grade PREREG-v0.1 from the hpimd.json files of the control and target jobs of each model.

    python grade_hpimd.py --pair mace-mpa-0 ctrl.json tgt.json --pair mattersim-5m ctrl.json tgt.json

A missing file is NOT GRADABLE for that model (never a pass or a fail).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpimd_core as hc  # noqa: E402


def classes(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        res = json.load(fh)
    return {r["tag"]: r["class"] for r in res.get("runs", [])}


def grade_model(ctrl_path, tgt_path):
    c, t = classes(ctrl_path), classes(tgt_path)
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
    ap.add_argument("--pair", nargs=3, action="append", metavar=("MODEL", "CTRL", "TGT"),
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
