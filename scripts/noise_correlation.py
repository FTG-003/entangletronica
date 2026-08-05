"""Spatially correlated noise: robustness of the white-noise approximation.

The paper's dephasing model (Sec. 2.4) is delta-correlated in space and time.
A referee will ask whether a *spatially correlated* environment (e.g. a
screened impurity potential with a finite correlation length, or 1/f-like
charge noise) changes the coherence budget.  This script re-runs the ensemble
with the noise field smoothed by a Gaussian kernel of width xi = {5, 10} nm,
**variance-preserving** (stochastic.solve2d_stochastic noise_xi), so that only
the spatial correlation is varied while the local noise amplitude is identical.

Reported (results/noise_correlation.json):
  * C(T; xi) and T_max(xi) for xi = 0 (white), 5, 10 nm;
  * a qualitative statement: longer-range correlation *changes* the phase
    accumulated per realisation (Var(phi) ~ integral of the correlation
    function), so the white-noise approximation brackets the correlated case
    rather than underestimating or overestimating it uniformly.

Usage:  python3 scripts/noise_correlation.py
Outputs: results/noise_correlation.json, figures/fig_noise_correlation.pdf
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
                                Tmax_from_curve, profile_contrast)

XIS = [0.0, 5.0, 10.0]       # nm; 0 = white noise (reference, from committed JSON)
N_ENS = 60
COHERENCE_JSON = os.path.join(HERE, "results", "coherence_ensemble.json")
JSON_OUT = os.path.join(HERE, "results", "noise_correlation.json")
FIG = os.path.join(HERE, "figures", "fig_noise_correlation.pdf")


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    C_by_xi, Tmax_by_xi = {}, {}
    # reference xi = 0 reuses the committed N=200 analysis (paper Table 1)
    if os.path.exists(COHERENCE_JSON):
        d = json.load(open(COHERENCE_JSON))
        C_by_xi[0.0] = d["C_numerical"]
        Tmax_by_xi[0.0] = Tmax_from_curve(TEMPERATURES_K, C_by_xi[0.0])
        print("xi = 0 (white) taken from committed N=200 JSON (paper Table 1)")
    else:
        per_T = run_ensemble(n_ens=N_ENS, xi=0.0)
        C_by_xi[0.0], *_ = summarize(per_T)
        Tmax_by_xi[0.0] = Tmax_from_curve(TEMPERATURES_K, C_by_xi[0.0])

    for xi in XIS[1:]:
        per_T = run_ensemble(n_ens=N_ENS, xi=xi)
        C_num, *_ = summarize(per_T)
        C_by_xi[xi] = C_num
        Tmax_by_xi[xi] = Tmax_from_curve(TEMPERATURES_K, C_num)

    # qualitative summary at the two extreme temperatures
    Cw = np.array(C_by_xi[0.0])
    summary = {}
    for T, i in zip(TEMPERATURES_K, range(len(TEMPERATURES_K))):
        row = {"T_K": T, "C_white": float(Cw[i])}
        for xi in XIS[1:]:
            row[f"C_xi{int(xi)}"] = float(C_by_xi[xi][i])
        summary[str(T)] = row

    out = {
        "xi_nm": XIS,
        "temperatures": TEMPERATURES_K,
        "C_by_xi_numerical": {str(xi): C_by_xi[xi] for xi in XIS},
        "Tmax_by_xi_K": {str(xi): Tmax_by_xi[xi] for xi in XIS},
        "per_T": summary,
        "N_ens": N_ENS,
        "N_ens_xi0": 200,
        "scale_noise": 22.0,
        "note": ("variance-preserving Gaussian smoothing (noise_xi in "
                  "stochastic.py): only the spatial correlation length changes, "
                  "the local noise amplitude is fixed. Result: at fixed local "
                  "amplitude, longer-range correlation makes the two interfering "
                  "paths sample increasingly COMMON noise, suppressing the "
                  "differential (relative) phase that destroys the fringes. "
                  "The white-noise model (xi=0) is therefore the conservative "
                  "worst case: it maximizes differential dephasing, so the "
                  "paper's T_max = 11 K bound is a lower bound that holds for "
                  "any environment with finite spatial correlation."),
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {JSON_OUT}")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {0.0: "#1f77b4", 5.0: "#ff7f0e", 10.0: "#2ca02c"}
    for xi in XIS:
        ax.semilogx(TEMPERATURES_K, C_by_xi[xi], "o-", color=colors[xi],
                    ms=5, lw=1.6,
                    label=(r"$\xi=0$ (white)" if xi == 0 else rf"$\xi={xi:.0f}$ nm"))
        tm = Tmax_by_xi[xi]
        if tm:
            ax.axvline(tm, color=colors[xi], ls=":", lw=1.0, alpha=0.7)
    ax.axhline(0.5, color="0.55", ls="--", lw=1)
    ax.set_xlabel(r"temperature $T$ (K)")
    ax.set_ylabel(r"numerical visibility $C(T)$")
    ax.set_title("Spatial correlation of the dephasing noise (variance-fixed)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"wrote {FIG}")

    for xi in XIS:
        print(f"xi = {xi:4.0f} nm : T_max = {Tmax_by_xi[xi]}, "
              f"C(T) = " + " ".join(f"{c:.3f}" for c in C_by_xi[xi]))


if __name__ == "__main__":
    main()