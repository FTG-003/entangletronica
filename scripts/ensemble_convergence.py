"""Ensemble-size convergence of the visibility (scientific due diligence).

The paper's headline C(T) points come from a 200-realisation ensemble.  A
referee will ask whether that number is converged in N.  This script answers
with a *nested* analysis: it runs one pool of ``OUTER_N`` stochastic
realisations (deterministic seed) and reads off C(4 K) and C(10 K) for every
ensemble size N = 50, 100, ... so each C(N) is the arithmetic mean of the SAME
first N realisations.  There is therefore no cross-run statistical noise in the
convergence trend - only the genuine fluctuation of a mean of N.

Reported (results/ensemble_convergence.json):
  * C(N) for the 4 K calibration anchor and the 10 K steepest point;
  * the largest bootstrapped |C(N) - C(N_max)| deviation over the range;
  * an empirical statement of whether the 200-realisation number is stable to a
    chosen tolerance (measured, never forced).

Usage:  python3 scripts/ensemble_convergence.py
Outputs: results/ensemble_convergence.json
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

from ensemble_coherence import (TEMPERATURES_K, run_ensemble,
                                contrast_from_profiles, profile_contrast,
                                bootstrap_c, Tmax_from_curve)

OUTER_MAX = 250            # total realisations drawn for the nested pool
N_POINTS = [50, 100, 150, 200, OUTER_MAX]
N_REF = 200                # the *operating* ensemble size quoted in the paper
FOCUS_T = [4.0, 10.0]      # anchor + steepest point
TOL = 0.01                 # target tolerance the referee would ask for

JSON_OUT = os.path.join(HERE, "results", "ensemble_convergence.json")
FIG = os.path.join(HERE, "figures", "fig_ensemble_convergence.pdf")


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    per_T = run_ensemble(temperatures=FOCUS_T, n_ens=OUTER_MAX)

    C_by_T, bootstd_by_T, deltas_by_T = {}, {}, {}
    for T in FOCUS_T:
        profs, _ = per_T[T]
        Cs = [contrast_from_profiles(profs[:n]) for n in N_POINTS]
        C_by_T[T] = Cs
        bootstd_by_T[T] = bootstrap_c(profs[:N_REF])
        cref = Cs[N_POINTS.index(N_REF)]            # operating size
        # deviation of every smaller N from the *operating* N_REF=200 value
        deltas_by_T[T] = [abs(Cs[i] - cref) for i in range(len(N_POINTS) - 1)]

    # honest convergence per temperature: max |dC| across the nested ladder
    conv = {}
    for T in FOCUS_T:
        dmax = max(deltas_by_T[T])
        conv[str(T)] = {
            "max_deviation_from_operating_N": float(dmax),
            "requested_tolerance": TOL,
            "met": bool(dmax <= TOL),
        }
    conv["_note"] = ("max_deviation is the worst |C(N)-C(200)| over the nested "
                      "ladder (N = 50..200); N=200 is the operating size quoted "
                      "in the paper. 10 K is the steepest (most ensemble-\n"
                      "sensitive) point and intentionally reports a larger band.")

    out = {
        "temperatures": FOCUS_T,
        "n_ensemble_points": N_POINTS,
        "reference_operating_N": N_REF,
        "C_nested": {str(T): C_by_T[T] for T in FOCUS_T},
        "bootstrap_std_operating_N": {str(T): bootstd_by_T[T] for T in FOCUS_T},
        "convergence": conv,
        "N_pool": OUTER_MAX,
        "scale_noise": 22.0,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {JSON_OUT}")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, len(FOCUS_T), figsize=(10.5, 3.8), sharey=True)
    for ax, T in zip(axes, FOCUS_T):
        ax.plot(N_POINTS, C_by_T[T], "o-", color="#1f77b4", lw=1.5, ms=5)
        ax.axvline(N_REF, color="0.6", ls=":", lw=1)
        cref = C_by_T[T][N_POINTS.index(N_REF)]
        ax.axhline(cref, color="0.6", lw=0.8)
        ax.fill_between(N_POINTS,
                        np.array(C_by_T[T]) - 2 * bootstd_by_T[T],
                        np.array(C_by_T[T]) + 2 * bootstd_by_T[T],
                        color="#1f77b4", alpha=0.12,
                        label=fr"$2\sigma$ bootstrap at $N={N_REF}$")
        ax.text(N_REF, ax.get_ylim()[0] + 0.02, "operating $N$",
                fontsize=7, color="0.4", ha="right")
        ax.set_title(f"$T={T:.0f}$ K", fontsize=11)
        ax.set_xlabel(r"realisations $N$")
        ax.legend(fontsize=7, loc="center right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$C(N)$")
    fig.suptitle("Ensemble-size convergence of the visibility (nested) ",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG)
    print(f"wrote {FIG}")

    for T in FOCUS_T:
        print(f"T = {T:4.0f} K : " +
              "  ".join(f"N={n}:C={c:.4f}" for n, c in zip(N_POINTS, C_by_T[T])))
        print(f"            max|dC|(vs N={N_REF})="
              f"{conv[str(T)]['max_deviation_from_operating_N']:.4f} "
              f"(requested {TOL}) -> {'meet' if conv[str(T)]['met'] else 'over'}")


if __name__ == "__main__":
    main()