#!/usr/bin/env python3
import os, time, json
import numpy as np

from sashimi_si import subhalo_properties


def run_sashimi_subhalo_catalog(
    M0: float,
    redshift: float,
    logmamin: int,
    sigma0_m: float,
    w: float,
    dz: float = 0.01,
    zmax: float = 5.0,
    outdir: str | None = None,
) -> dict:
    """
    Runs SASHIMI-SI and saves:
      - catalog_raw.npz
      - catalog_SIDM_survivors.npz
      - summary.json

    Returns a dict with paths + summary.
    """

    # --- setup output dir ---
    if outdir is None:
        tag = time.strftime("run_%Y%m%d_%H%M%S")
        outdir = f"/Users/sdasbakshi/Documents/cosmowork/sashimi_si_project/runs/{tag}"
    os.makedirs(outdir, exist_ok=True)

    # --- run model ---
    sh = subhalo_properties(sigma0_m=sigma0_m, w=w)
    out = sh.subhalo_properties_calc(
        M0=M0,
        redshift=redshift,
        dz=dz,
        zmax=zmax,
        logmamin=logmamin,
    )

    names = [
        "ma200","z_acc","rsCDM_acc","rhosCDM_acc","rmaxCDM_acc","VmaxCDM_acc",
        "rsSIDM_acc","rhosSIDM_acc","rcSIDM_acc","rmaxSIDM_acc","VmaxSIDM_acc",
        "m_z0","rsCDM_z0","rhosCDM_z0","rmaxCDM_z0","VmaxCDM_z0",
        "rsSIDM_z0","rhosSIDM_z0","rcSIDM_z0","rmaxSIDM_z0","VmaxSIDM_z0",
        "ctCDM_z0","tt_ratio","weightCDM","weightSIDM","surviveCDM","surviveSIDM"
    ]

    if len(out) != len(names):
        raise RuntimeError(f"Unexpected output length: len(out)={len(out)} but expected {len(names)}")

    col = {n: out[i] for i, n in enumerate(names)}

    # --- filtering: SIDM survivors + physical cuts ---
    mask = col["surviveSIDM"].astype(bool)
    mask &= (col["m_z0"] > 0)
    mask &= (col["rmaxSIDM_z0"] > 0)
    mask &= (col["VmaxSIDM_z0"] > 0)
    mask &= (col["rsSIDM_z0"] > 0)
    mask &= (col["rhosSIDM_z0"] > 0)
    mask &= (col["rhosSIDM_acc"] > 0)
    mask &= (col["rmaxSIDM_acc"] > 0)
    mask &= (col["VmaxSIDM_acc"] > 0)
    mask &= (col["rsSIDM_acc"] > 0)

    col_f = {k: v[mask] for k, v in col.items()}

    # --- metadata + summary ---
    meta = dict(
        M0=float(M0),
        redshift=float(redshift),
        logmamin=int(logmamin),
        sigma0_m=float(sigma0_m),
        w=float(w),
        dz=float(dz),
        zmax=float(zmax),
    )

    summary = dict(
        **meta,
        N_total=int(len(col["m_z0"])),
        N_survive=int(np.sum(col["surviveSIDM"] == 1)),
        N_filtered=int(len(col_f["m_z0"])),
        frac_filtered=float(len(col_f["m_z0"]) / len(col["m_z0"])),
        m_z0_quantiles=np.quantile(col_f["m_z0"], [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
        columns=list(col_f.keys()),
    )

    raw_path = os.path.join(outdir, "catalog_raw.npz")
    filt_path = os.path.join(outdir, "catalog_SIDM_survivors.npz")
    summ_path = os.path.join(outdir, "summary.json")

    np.savez_compressed(raw_path, **col, meta=np.array([meta], dtype=object))
    np.savez_compressed(filt_path, **col_f)

    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)

    return dict(outdir=outdir, raw=raw_path, filtered=filt_path, summary=summ_path, summary_obj=summary)


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Run SASHIMI-SI subhalo catalog forward model.")
    p.add_argument("--M0", type=float, required=True, help="Host halo mass at redshift (Msun).")
    p.add_argument("--redshift", type=float, default=0.0)
    p.add_argument("--logmamin", type=int, default=6)
    p.add_argument("--sigma0_m", type=float, required=True)
    p.add_argument("--w", type=float, required=True)
    p.add_argument("--dz", type=float, default=0.01)
    p.add_argument("--zmax", type=float, default=5.0)
    p.add_argument("--outdir", type=str, default=None, help="Output directory; if omitted, auto-tagged.")
    args = p.parse_args()

    res = run_sashimi_subhalo_catalog(
        M0=args.M0,
        redshift=args.redshift,
        logmamin=args.logmamin,
        sigma0_m=args.sigma0_m,
        w=args.w,
        dz=args.dz,
        zmax=args.zmax,
        outdir=args.outdir,
    )

    # Print JSON to stdout so ArgoLOOM can parse deterministically
    print(json.dumps({
        "outdir": res["outdir"],
        "raw": res["raw"],
        "filtered": res["filtered"],
        "summary": res["summary"],
        "summary_obj": res["summary_obj"],
    }))

