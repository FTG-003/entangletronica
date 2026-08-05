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
def _job_seed(tidx, r, scale, xi):
    """Deterministic per-realisation seed (independent across scale / xi runs).

    For the operating point (scale=22, xi=0) this reduces exactly to the
    original formula ``SEED + tidx * 1_000_000 + r``, so the committed
    results/coherence_ensemble.json reproduces byte-for-byte.
    """
    off = ((int(round(scale * 10.0)) - 220) * 100_003
           + int(round(xi * 10.0)) * 50_017)
    return SEED + tidx * 1_000_000 + r + off


def _job(args):
    """One stochastic realisation: (T_idx, realisation_idx, scale, xi)."""
    tidx, r, scale, xi = args
    rng = np.random.default_rng(_job_seed(tidx, r, scale, xi))
    psi, _ = stochastic.solve2d_stochastic(
        V, psi0, DT, NT, X, Y, tau_phi_nat[tidx], rng,
        scale_noise=scale, noise_xi=xi)
    p = np.abs(psi) ** 2
    prof = p[IDET, :]
    prof = prof / prof.sum()
    imb, _, _ = detector_imbalance(psi, xdet=DET_X)
    return prof, imb


def run_ensemble(temperatures=None, n_ens=N_ENS, scale=SCALE_NOISE, xi=0.0,
                 nproc=4):
    """Ensemble of stochastic realisations; returns ``{T: (profs, imbs)}``.

    Deterministic for a fixed (n_ens, scale, xi): each realisation uses its own
    seeded Generator, so the result is identical under any process scheduling.
    """
    from multiprocessing import Pool
    if temperatures is None:
        temperatures = list(TEMPERATURES_K)
    tidx_of = {T: i for i, T in enumerate(TEMPERATURES_K)}
    t0 = time.time()
    tasks = [(tidx_of[T], r, scale, xi)
             for T in temperatures for r in range(n_ens)]
    with Pool(nproc) as pool:
        results = pool.map(_job, tasks)
    per_T = {}
    for k, T in enumerate(temperatures):
        chunk = results[k * n_ens:(k + 1) * n_ens]
        per_T[T] = (np.array([c[0] for c in chunk]),
                    np.array([c[1] for c in chunk]))
    print(f"ensemble {len(temperatures) * n_ens} solves in "
          f"{time.time() - t0:.0f} s", flush=True)
    return per_T


def contrast_from_profiles(profs):
    """C = (max-min)/(max+min) of the ensemble-mean detector profile."""
    return float(profile_contrast(profs.mean(axis=0)))


def Tmax_from_curve(T_list, C_list, half=C_HALF):
    """Linear interpolation of the first C = half crossing of C(T)."""
    for k in range(len(T_list) - 1):
        if (C_list[k] - half) * (C_list[k + 1] - half) < 0:
            t1, t2 = T_list[k], T_list[k + 1]
            c1, c2 = C_list[k], C_list[k + 1]
            return float(t1 + (half - c1) * (t2 - t1) / (c2 - c1))
    return None


def bootstrap_c(profs, n=N_BOOT, rng_seed=7):
    """Std of C(mean profile) under resampling the realisations."""
    rng = np.random.default_rng(rng_seed)
    n = profs.shape[0]
    Cs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Cs[b] = profile_contrast(profs[idx].mean(axis=0))
    return float(Cs.std())


def summarize(per_T, T_list=None):
    """Contrasts / imbalances / analytical curve for a per-T ensemble dict."""
    if T_list is None:
        T_list = list(per_T)
    C_num, C_std, imb_mean, imb_std, imb_profile = [], [], [], [], []
    for T in T_list:
        profs, imbs = per_T[T]
        mean_prof = profs.mean(axis=0)
        C_num.append(float(profile_contrast(mean_prof)))
        C_std.append(bootstrap_c(profs))
        imb_mean.append(float(imbs.mean()))
        imb_std.append(float(imbs.std()))
        pu = mean_prof[(Y >= 0) & (Y < 14.0)].sum()
        pl = mean_prof[(Y > -14.0) & (Y < 0)].sum()
        imb_profile.append(float((pu - pl) / (pu + pl)))
    C_ana = [analytical_contrast(T) for T in T_list]
    return C_num, C_std, imb_mean, imb_std, imb_profile, C_ana


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    per_T = run_ensemble()

    C_num, C_std, imb_mean, imb_std, imb_profile, C_ana = summarize(per_T)
    profiles_mean = [per_T[T][0].mean(axis=0) for T in TEMPERATURES_K]
    Tc = Tmax_from_curve(TEMPERATURES_K, C_num)

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
