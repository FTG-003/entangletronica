"""Scale-factor sensitivity of the coherence budget (scientific due diligence).

The noise amplitude is calibrated empirically (s = 22) to pin C(4 K) to the
analytical anchor.  A referee will ask how much the headline operating bound
T_max(C=0.5) moves if the calibration is off.  This script answers by
re-running the ensemble at s in {22-5, 22, 22+5} = {17, 22, 27} at the SAME
resolution as the paper (N_ens = 200, deterministic per-realisation seeds):

  * the operating curve s = 22 is taken from the committed
    results/coherence_ensemble.json (N = 200) so T_max(22) is exactly the
    paper value 11.4 K;
  * s = 17 and s = 27 are computed fresh at N = 200.

Reported (results/scale_sensitivity.json):
  * C(T; s) for every temperature in [4, 10, 20, 50, 77] K;
  * T_max(s) by the same linear crossing as the main study;
  * the local sensitivity d T_max / d s and the implied T_max uncertainty from
    a calibration error delta(s) = +/- 5.

This is a robustness statement, not a re-calibration: s = 22 stays the
reported operating value.

Usage:  python3 scripts/scale_sensitivity.py
Outputs: results/scale_sensitivity.json, figures/fig_scale_sensitivity.pdf
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from ensemble_coherence import (TEMPERATURES_K, run_ensemble, summarize,
                                Tmax_from_curve)

SCALES = [17.0, 22.0, 27.0]     # s = 22 -/+ 5
N_ENS = 200
DS_CAL = 5.0                    # assumed calibration uncertainty in s
COHERENCE_JSON = os.path.join(HERE, "results", "coherence_ensemble.json")
JSON_OUT = os.path.join(HERE, "results", "scale_sensitivity.json")
FIG = os.path.join(HERE, "figures", "fig_scale_sensitivity.pdf")


def load_operating_C():
    """Operating C(s=22) from the committed N=200 analysis (paper Table 1)."""
    import json as _json
    with open(COHERENCE_JSON) as f:
        d = _json.load(f)
    return list(d["C_numerical"]), list(d["temperatures"])


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    C_by_s, Tmax_by_s = {}, {}
    # operating point reuses the committed N=200 analysis -> exact paper value
    if os.path.exists(COHERENCE_JSON):
        d = json.load(open(COHERENCE_JSON))
        C_by_s[22.0] = d["C_numerical"]
        Tmax_by_s[22.0] = Tmax_from_curve(TEMPERATURES_K, C_by_s[22.0])
        print("s = 22 (operating) taken from committed N=200 JSON "
              "(paper Table 1)")
    else:
        per_T = run_ensemble(n_ens=N_ENS, scale=22.0)
        C_by_s[22.0], *_ = summarize(per_T)
        Tmax_by_s[22.0] = Tmax_from_curve(TEMPERATURES_K, C_by_s[22.0])

    # sensitivity wings re-run at the same N=200 resolution
    for s in [SCALES[0], SCALES[2]]:
        per_T = run_ensemble(n_ens=N_ENS, scale=s)
        C_num, *_ = summarize(per_T)
        C_by_s[s] = C_num
        Tmax_by_s[s] = Tmax_from_curve(TEMPERATURES_K, C_num)

    # two-sided finite-difference sensitivity of T_max wrt s
    d = (Tmax_by_s[SCALES[2]] - Tmax_by_s[SCALES[0]]) / (SCALES[2] - SCALES[0])
    dT = DS_CAL * abs(d)

    out = {
        "scales": SCALES,
        "temperatures": TEMPERATURES_K,
        "C_by_scale_numerical": {str(s): C_by_s[s] for s in SCALES},
        "Tmax_by_scale_K": {str(s): Tmax_by_s[s] for s in SCALES},
        "Tmax_reference_K": Tmax_by_s[22.0],
        "dTmax_ds_K_per_unit_s": float(d),
        "Tmax_uncertainty_from_scale_delta5_K": float(dT),
        "N_ens": N_ENS,
        "scale_operating": 22.0,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {JSON_OUT}")

    # ---------------------------------------------------------------- figure
    fig, (ax, axi) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    colors = {17.0: "#2ca02c", 22.0: "#d62728", 27.0: "#9467bd"}
    for s in SCALES:
        ax.semilogx(TEMPERATURES_K, C_by_s[s], "o-", color=colors[s], ms=5,
                    lw=1.6, label=fr"$s={s:.0f}$")
        tm = Tmax_by_s[s]
        if tm:
            ax.axvline(tm, color=colors[s], ls=":", lw=1.0, alpha=0.7)
    ax.axhline(0.5, color="0.55", ls="--", lw=1)
    ax.text(55, 0.52, r"$C=0.5$", color="0.4", fontsize=8, va="bottom")
    ax.set_xlabel(r"temperature $T$ (K)")
    ax.set_ylabel(r"numerical visibility $C(T;s)$")
    ax.set_title("Scale-factor sensitivity of the coherence budget", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    xs = list(SCALES)
    ys = [Tmax_by_s[s] for s in SCALES]
    axi.plot(xs, ys, "o-", color="#1f77b4", lw=1.6, ms=6)
    axi.set_xlabel(r"noise scale $s$")
    axi.set_ylabel(r"$T_{\mathrm{max}}(C=0.5)$ (K)")
    axi.set_title(fr"$dT_{{\rm max}}/ds={d:+.2f}$ K/unit", fontsize=9)
    axi.grid(alpha=0.3)
    axi.axhline(11.0, color="0.6", ls=":", lw=1)

    fig.tight_layout()
    fig.savefig(FIG)
    print(f"wrote {FIG}")

    for s in SCALES:
        print(f"s = {s:4.0f} : T_max = {Tmax_by_s[s]:6.2f} K, "
              f"C(T) = " + " ".join(f"{c:.2f}" for c in C_by_s[s]))
    print(f"dT_max/ds = {d:+.3f} K/unit  ->  T_max = "
          f"{Tmax_by_s[22.0]:.1f} \u00b1 {dT:.1f} K  (s = 22 \u00b1 {DS_CAL:.0f})")


if __name__ == "__main__":
    main()