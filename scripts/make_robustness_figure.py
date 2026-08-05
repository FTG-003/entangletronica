#!/usr/bin/env python3
"""Four-panel referee due-diligence figure (paper Fig.~robust).

Panels:
  (a) Noiseless limit: the stochastic solver with scale_noise = 0 must
      reproduce the deterministic run exactly (max |dpsi| = 0) and the
      full-line contrast must tend to C_det ~ 1.0.
  (b) Ensemble-size convergence (results/ensemble_convergence.json):
      C(4 K) flat to +-0.01 vs operating N = 200; the 10 K point reports
      its wider honest band.
  (c) Noise-scale sensitivity (results/scale_sensitivity.json):
      T_max(s) with dT_max/ds ~ -0.73 K/unit.
  (d) Spatially correlated noise (results/noise_correlation.json):
      C(T) for xi = 0, 5, 10 nm at fixed local amplitude; T_max rises.

Run:  python3 scripts/make_robustness_figure.py
Out:  figures/fig_robustness.pdf
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from entangletronica import electron, stochastic, potential as P

FIG = os.path.join(ROOT, "figures", "fig_robustness.pdf")
RES = os.path.join(ROOT, "results")

# operating configuration (scripts/entangletron_experiment.py)
NX, NY, DX = 140, 80, 2.0
X = np.arange(NX) * DX - 40.0
Y = np.arange(NY) * DX - 80.0
xx, yy = np.meshgrid(X, Y, indexing="ij")
K0, S = 0.2, 10.0
DT, NT = 0.30, 1300
DET_X = 110.0


def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def panel_noiseless(ax):
    psi0 = electron.gaussian_packet(xx, yy, k0=K0, s=S)
    V = P.young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT
    psi_det, _, _ = electron.solve2d(V, psi0, DT, NT, X, Y, report=False)
    rng = np.random.default_rng(20260201)
    psi_st, norm = stochastic.solve2d_stochastic(
        V, psi0, DT, NT, X, Y, 1.0, rng, scale_noise=0.0)
    dmax = float(np.max(np.abs(psi_st - psi_det)))
    i = int(np.argmin(np.abs(X - DET_X)))
    p_det = np.abs(psi_det[i, :]) ** 2
    p_det = p_det / p_det.sum()
    C_det = (p_det.max() - p_det.min()) / (p_det.max() + p_det.min())
    ax.plot(Y, p_det, lw=1.4, color="#1f77b4", label="deterministic")
    ax.plot(Y, np.abs(psi_st[i, :]) ** 2 / (np.abs(psi_st[i, :]) ** 2).sum(),
            lw=0.8, ls="--", color="#ff7f0e", label="stochastic, $s_\\phi=0$")
    ax.text(0.97, 0.93,
            f"max $|\\Delta\\psi|=10^{{{int(np.log10(dmax + 1e-300)):.0f}}}$\n"
            f"$C_{{\\rm det}}$ = {C_det:.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
    ax.set_xlabel("$y$ (nm)")
    ax.set_ylabel("$P(y)$ (norm.)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) noiseless limit", fontsize=10)


def panel_convergence(ax):
    d = _load("ensemble_convergence.json")
    Ns = d["n_ensemble_points"]
    for Tk in ["4.0", "10.0"]:
        C = d["C_nested"][Tk]
        ax.plot(Ns, C, marker="o", ms=4, lw=1.2,
                label=f"$T={float(Tk):.0f}\\,$K", color=("#1f77b4" if Tk == "4.0" else "#d62728"))
    c4 = d["C_nested"]["4.0"][Ns.index(200)]
    ax.axhline(c4 + 0.01, ls=":", lw=0.8, color="#1f77b4")
    ax.axhline(c4 - 0.01, ls=":", lw=0.8, color="#1f77b4")
    ax.axvline(200, ls="--", lw=0.8, color="k")
    ax.text(206, 0.96, "operating $N$", fontsize=8)
    ax.annotate("$\\pm0.01$ band",
                xy=(140, c4 + 0.012), fontsize=8, color="#1f77b4")
    ax.set_xlabel("$N_{\\rm ens}$")
    ax.set_ylabel("$\\langle C\\rangle$")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(b) ensemble-size convergence", fontsize=10)


def panel_scale(ax):
    d = _load("scale_sensitivity.json")
    s = np.array(d["scales"])
    tm = np.array([d["Tmax_by_scale_K"][f"{x:.1f}"] for x in s])
    ax.plot(s, tm, marker="o", ms=5, lw=1.4, color="#2ca02c")
    slope = d["dTmax_ds_K_per_unit_s"]
    sf = np.linspace(s.min(), s.max(), 50)
    ax.plot(sf, d["Tmax_by_scale_K"]["22.0"] + slope * (sf - 22.0),
            ls="--", lw=0.8, color="#2ca02c")
    ax.fill_between([17, 27], 11.4 - 3.64, 11.4 + 3.64, alpha=0.12, color="#2ca02c")
    ax.text(22.4, 20.5,
            f"$\\mathrm{{d}}T_{{\\max}}/\\mathrm{{d}}s_\\phi$ = {slope:.2f} K\n"
            f"$T_{{\\max}}=11.4\\pm3.6$ K  ($s_\\phi=22\\pm5$)",
            fontsize=9)
    ax.set_xlabel("noise scale $s_\\phi$")
    ax.set_ylabel("$T_{\\max}$ (K)")
    ax.set_title("(c) noise-scale sensitivity", fontsize=10)


def panel_correlation(ax):
    d = _load("noise_correlation.json")
    T = np.array(d["temperatures"])
    colors = {"0.0": "#1f77b4", "5.0": "#ff7f0e", "10.0": "#2ca02c"}
    for xi in ["0.0", "5.0", "10.0"]:
        C = d["C_by_xi_numerical"][xi]
        ax.plot(T, C, marker="o", ms=4, lw=1.3, color=colors[xi],
                label=f"$\\xi={float(xi):.0f}$ nm")
    tmx = d["Tmax_by_xi_K"]
    for xi in ["0.0", "5.0", "10.0"]:
        ax.axvline(tmx[xi], ls=":", lw=0.8, color=colors[xi])
    ax.text(11.9, 0.9, "$T_{\\max}$: 11.4 $\\to$ 14.8 $\\to$ 29.2 K",
            fontsize=9)
    ax.set_xlabel("$T$ (K)")
    ax.set_ylabel("$\\langle C\\rangle$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(d) spatially correlated noise", fontsize=10)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    panel_noiseless(axes[0, 0])
    panel_convergence(axes[0, 1])
    panel_scale(axes[1, 0])
    panel_correlation(axes[1, 1])
    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
