#!/usr/bin/env python3
"""
API_SASHIMI_subhalos.py

ArgoLOOM tool wrapper: generate SASHIMI-SI subhalo catalog.

Input:
- JSON via env var ARGOLOOM_SASHIMI_PARAMS_JSON with keys:
    M0, sigma0_m, w (required)
    redshift, logmamin, dz, zmax, outdir (optional)

Output:
- Prints JSON to stdout with paths + summary.
"""

import os
import sys
import json
import subprocess


SASHIMI_ADAPTER = "/Users/sdasbakshi/Documents/cosmowork/sashimi_si_project/argoloom_sashimi_adapter.py"


def main() -> int:
    params_json = os.environ.get("ARGOLOOM_SASHIMI_PARAMS_JSON", "").strip()
    if not params_json:
        # Safe default (smoke run) if called directly
        params = dict(M0=1e12, redshift=0.0, logmamin=6, sigma0_m=147.1, w=24.33)
    else:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[SASHIMI] ERROR: invalid ARGOLOOM_SASHIMI_PARAMS_JSON: {e}\n")
            return 2

    # Basic required fields
    for k in ["M0", "sigma0_m", "w"]:
        if k not in params:
            sys.stderr.write(f"[SASHIMI] ERROR: missing required param '{k}' in params: {params}\n")
            return 2

    cmd = [
        sys.executable, SASHIMI_ADAPTER,
        "--M0", str(params["M0"]),
        "--sigma0_m", str(params["sigma0_m"]),
        "--w", str(params["w"]),
    ]

    # Optional args
    if "redshift" in params: cmd += ["--redshift", str(params["redshift"])]
    if "logmamin" in params: cmd += ["--logmamin", str(params["logmamin"])]
    if "dz" in params:       cmd += ["--dz", str(params["dz"])]
    if "zmax" in params:     cmd += ["--zmax", str(params["zmax"])]
    if "outdir" in params and params["outdir"]:
        cmd += ["--outdir", str(params["outdir"])]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        sys.stderr.write("[SASHIMI] Adapter failed\n")
        sys.stderr.write(f"CMD: {' '.join(cmd)}\n")
        sys.stderr.write("STDOUT:\n" + proc.stdout + "\n")
        sys.stderr.write("STDERR:\n" + proc.stderr + "\n")
        return proc.returncode

    # Adapter prints JSON on stdout; just forward it.
    out = proc.stdout.strip()
    # Validate it is JSON (so ArgoLOOM doesn't ingest garbage)
    try:
        _ = json.loads(out)
    except json.JSONDecodeError:
        sys.stderr.write("[SASHIMI] ERROR: adapter stdout is not valid JSON\n")
        sys.stderr.write("STDOUT:\n" + proc.stdout + "\n")
        sys.stderr.write("STDERR:\n" + proc.stderr + "\n")
        return 3

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

