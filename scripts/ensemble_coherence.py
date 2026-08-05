"""Ensemble coherence study: stochastic dephasing vs temperature (Sec. 2.4).

For each temperature T in [4, 10, 20, 50, 77] K runs N_ens = 200 realisations of
the split-step solver with delta-correlated (white-in-space, white-in-time)
potential noise, extracts the detector profile at x = 110 nm, and reports:

    C_numerical(T)   visibility (max-min)/(max+min) of the ensemble-mean profile
    C_analytical(T)  0.95 * exp(-t_transit / tau_phi(T))   [phenomenological]
    imbalance         (P_U - P_L)/(P_U + P_L), bins |y| <= 14 nm

The noise amplitude scale_noise is calibrated empirically so that the numerical
4 K visibility matches the analytical curve (see the methodological note in the
paper): the delta-correlated model overestimates the low-T dephasing when
normalised naively, because the two interfering paths overlap spatially for
most of the transit and only the *differential* phase matters.

Results:   results/coherence_ensemble.json
Figure:    figures/fig_coherence_ensemble.pdf

Usage:  python3 scripts/ensemble_coherence.py
"""

import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from entangletronica import potential as P
from entangletronica import stochastic
from entangletron_experiment import (X, Y, xx, yy, psi0, DT, NT,
                                     young_landscape, detector_imbalance, DET_X)

# ------------------------------------------------------------------ parameters
TEMPERATURES_K = [4.0, 10.0, 20.0, 50.0, 77.0]
N_ENS = 200
TAU0_PS = 12.0          # dephasing time at T0
T0_K = 4.0
P_EXP = 1.5             # tau_phi(T) = TAU0 * (T0/T)**1.5
T_TRANSIT_PS = 0.24     # transit time across the interferometer (paper Sec. 2.4)
SCALE_NOISE = 22.0      # empirically calibrated (4 K point, see note)
C0_ANALYTICAL = 0.95    # ideal (noiseless) fringe visibility in the paper model
C_HALF = 0.5            # reference line in the figure
N_BOOT = 200            # bootstrap resamples for the C errorbar / 1-sigma band
SEED = 20260201

FIG = os.path.join(HERE, "figures", "fig_coherence_ensemble.pdf")
JSON_OUT = os.path.join(HERE, "results", "coherence_ensemble.json")

IDET = int(np.argmin(np.abs(X - DET_X)))
CUTOFF = None           # full detector line (task definition, tails included)

tau_phi_ps = [TAU0_PS * (T0_K / T) ** P_EXP for T in TEMPERATURES_K]
tau_phi_nat = [tau * P.PS_TO_NAT for tau in tau_phi_ps]
V = young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT


def profile_contrast(prof):
    """C = (max - min)/(max + min) of a detector profile (task definition)."""
    if CUTOFF is not None:
        prof = prof[np.abs(Y) <= CUTOFF]
    mx, mn = prof.max(), prof.min()
    return (mx - mn) / (mx + mn) if mx + mn > 0 else 0.0


def analytical_contrast(T):
    tau = TAU0_PS * (T0_K / T) ** P_EXP
    return C0_ANALYTICAL * np.exp(-T_TRANSIT_PS / tau)


# ------------------------------------------------------------------ ensemble
def _job(args):
    """One stochastic realisation: (T_idx, realization_idx) -> (profile, imb)."""
    tidx, r = args
    rng = np.random.default_rng(SEED + tidx * 1_000_000 + r)
    psi, norm = stochastic.solve2d_stochastic(
        V, psi0, DT, NT, X, Y, tau_phi_nat[tidx], rng,
        scale_noise=SCALE_NOISE)
    p = np.abs(psi) ** 2
    prof = p[IDET, :]
    prof = prof / prof.sum()
    imb, _, _ = detector_imbalance(psi, xdet=DET_X)
    return prof, imb


def run_ensemble(nproc=4):
    from multiprocessing import Pool
    t0 = time.time()
    tasks = [(t, r) for t in range(len(TEMPERATURES_K)) for r in range(N_ENS)]
    with Pool(nproc) as pool:
        results = pool.map(_job, tasks)
    per_T = {}
    for t in range(len(TEMPERATURES_K)):
        profs = np.array([results[t * N_ENS + r][0] for r in range(N_ENS)])
        imbs = np.array([results[t * N_ENS + r][1] for r in range(N_ENS)])
        per_T[t] = (profs, imbs)
    print(f"ensemble {len(TEMPERATURES_K) * N_ENS} solves in "
          f"{time.time() - t0:.0f} s", flush=True)
    return per_T


def bootstrap_c(profs, n=N_BOOT, rng_seed=7):
    """Std of C(mean profile) under resampling the realisations."""
    rng = np.random.default_rng(rng_seed)
    n = profs.shape[0]
    Cs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Cs[b] = profile_contrast(profs[idx].mean(axis=0))
    return float(Cs.std())


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    per_T = run_ensemble()

    C_num, C_std = [], []
    imb_mean, imb_std, imb_profile = [], [], []
    profiles_mean = []
    for t in range(len(TEMPERATURES_K)):
        profs, imbs = per_T[t]
        mean_prof = profs.mean(axis=0)
        profiles_mean.append(mean_prof)
        C_num.append(float(profile_contrast(mean_prof)))
        C_std.append(bootstrap_c(profs))
        imb_mean.append(float(imbs.mean()))
        imb_std.append(float(imbs.std()))
        pu = mean_prof[(Y >= 0) & (Y < 14.0)].sum()
        pl = mean_prof[(Y > -14.0) & (Y < 0)].sum()
        imb_profile.append(float((pu - pl) / (pu + pl)))

    C_ana = [analytical_contrast(T) for T in TEMPERATURES_K]

    # temperature where the numerical visibility crosses C = 0.5
    Tc = np.nan
    for k in range(len(TEMPERATURES_K) - 1):
        if (C_num[k] - C_HALF) * (C_num[k + 1] - C_HALF) < 0:
            t1, t2 = TEMPERATURES_K[k], TEMPERATURES_K[k + 1]
            c1, c2 = C_num[k], C_num[k + 1]
            Tc = t1 + (C_HALF - c1) * (t2 - t1) / (c2 - c1)
            break

    out = {
        "temperatures": TEMPERATURES_K,
        "C_numerical": C_num,
        "C_analytical": C_ana,
        "C_std": C_std,
        "imbalance_mean": imb_mean,
        "imbalance_std": imb_std,
        "imbalance_mean_profile": imb_profile,
        "N_ens": N_ENS,
        "scale_noise": SCALE_NOISE,
        "tau_phi_ps": tau_phi_ps,
        "t_transit_ps": T_TRANSIT_PS,
        "T_cross_half_K": None if np.isnan(Tc) else float(Tc),
        "note": ("delta-correlated white noise; scale_noise empirically "
                 "calibrated so C(4 K) matches C_ana(4 K) (see scripts/"
                 "_calibrate_scale.py); high-T deviation from the exponential "
                 "reflects the strong-scattering limit of the white-noise "
                 "model (paper Sec. 2.4, methodological note)"),
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {JSON_OUT}")

    # ------------------------------------------------------------------ figure
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    Ts = np.logspace(np.log10(4), np.log10(77), 200)
    Cana_curve = [analytical_contrast(T) for T in Ts]
    ax.semilogx(Ts, Cana_curve, "-", color="#1f77b4", lw=1.8,
                label=r"$C_{\mathrm{ana}}(T)=0.95\,e^{-t_{\rm tr}/\tau_\phi(T)}$")
    C_num = np.array(C_num)
    C_std = np.array(C_std)
    ax.errorbar(TEMPERATURES_K, C_num, yerr=C_std, fmt="o", ms=6, capsize=3,
                color="#d62728", ecolor="#d62728", elinewidth=1.4,
                label="numerical ensemble")
    ax.fill_between(TEMPERATURES_K, C_num - C_std, C_num + C_std,
                    color="#d62728", alpha=0.18, label=r"$1\sigma$ ensemble band")
    ax.axhline(C_HALF, color="0.55", lw=1, ls="--")
    ax.text(60, 0.52, r"$C=0.5$", fontsize=8, color="0.4", va="bottom")
    if not np.isnan(Tc):
        ax.axvline(Tc, color="0.55", lw=1, ls=":")
        ax.annotate(rf"$T_{{\rm max}}\approx{Tc:.0f}$ K",
                    xy=(Tc, C_HALF), xytext=(Tc * 1.35, 0.30),
                    fontsize=9, color="0.35",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9))
    ax.set_xlabel("temperature $T$ (K)")
    ax.set_ylabel(r"fringe visibility $C(T)$")
    ax.set_title(r"Coherence loss $\propto e^{-t_{\rm tr}/\tau_\phi(T)}$",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)

    # mean detector profiles
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(TEMPERATURES_K)))
    for t, T in enumerate(TEMPERATURES_K):
        axp.plot(Y, profiles_mean[t], color=colors[t], lw=1.4,
                 label=f"$T={T:.0f}$ K")
    axp.set_xlabel(r"$y$ (nm)")
    axp.set_ylabel(r"$\langle P(y)\rangle$ (norm.)")
    axp.set_title("ensemble-mean detector profile", fontsize=10)
    axp.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axp.grid(alpha=0.3)
    axp.set_xlim(-45, 45)

    fig.tight_layout()
    fig.savefig(FIG)
    print(f"wrote {FIG}")

    for T, c, ca, s in zip(TEMPERATURES_K, C_num, C_ana, C_std):
        print(f"T={T:5.1f}K  C_num={c:.3f}  C_ana={ca:.3f}  "
              f"1sigma={s:.3f}  I={imb_mean[TEMPERATURES_K.index(T)]:+.3f}")
    if not np.isnan(Tc):
        print(f"T_max (C=0.5 crossing) = {Tc:.1f} K")


if __name__ == "__main__":
    main()
