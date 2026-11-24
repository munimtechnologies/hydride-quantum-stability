# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Munim, Inc. (MV2). Made with AI agents working in MV2's name.
"""pin_hpimd 0.1.1 -- fetch the frozen, public hpimd 0.1.0 files that 0.1.1 keeps unchanged.

AMENDMENT-HPIMD-0.2.md: hpimd 0.1.1 changes only the job driver (hpimd_sim.py) and the grader
(grade_hpimd.py). The integrator, structures, metrics, thresholds and their unit tests
(hpimd_core.py, test_hpimd_core.py) and the pre-registration (PREREG-v0.1.md) are downloaded
from the public repository at a fixed commit and checked against their git blob sha1 (FATAL on
any mismatch), so nothing frozen is retyped.

    python pin_hpimd.py --dest /work/src
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

VERSION = "0.1.1"
REPO_RAW = "https://raw.githubusercontent.com/munimtechnologies/hydride-quantum-stability/"
COMMIT = "8f76b8240a4d8807cae6102e0ca46dd2d4bcf4b7"
# git blob sha1 of each file at COMMIT (repository tree API, read 2026-11-24)
BLOBS = {
    "hpimd_core.py": "eeec0fef77d56030557e0d9ed794e799b0f97b8f",
    "test_hpimd_core.py": "401aba0087618dfe91e74a983a09e662f6f53a34",
    "PREREG-v0.1.md": "60dbe6391d12775a58ed25282acb93a2447be0ac",
}


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(name):
    url = REPO_RAW + COMMIT + "/" + name
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pin_hpimd/" + VERSION})
            with urllib.request.urlopen(req, timeout=120) as fh:
                return fh.read()
        except Exception as exc:  # pragma: no cover - network
            print("download attempt %d of %s failed: %s" % (k + 1, url, exc), flush=True)
            time.sleep(10 * (k + 1))
    raise SystemExit("FATAL could not download " + url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    print("PIN pin_hpimd %s commit %s" % (VERSION, COMMIT), flush=True)
    for name, blob in BLOBS.items():
        raw = fetch(name)
        got = git_blob_sha1(raw)
        if got != blob:
            raise SystemExit("FATAL %s blob %s != pinned %s" % (name, got, blob))
        with open(os.path.join(args.dest, name), "wb") as fh:
            fh.write(raw)
        print("PIN_OK %s blob=%s sha256=%s" % (name, got, hashlib.sha256(raw).hexdigest()),
              flush=True)
    print("PIN_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
