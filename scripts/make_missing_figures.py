"""Generate the three figures referenced by EQLI_PhaseGate_Benchmark_2026.tex that the
original pipeline did not produce:

    fig_poisson_mapping.pdf  -- gate-voltage to effective-potential mapping
                                (Poisson + Thomas-Fermi screening)
    fig_coherence.pdf        -- ensemble visibility vs temperature
    fig_xor_schematic.pdf    -- two-input XOR gate proposal schematic

All numbers follow the paper text:  InGaAs 2DEG, m* = 0.042 m0,
eps_r = 13.9, n_2D = 2e11 cm^-2, V_g sweep -0.5..0 V, gate finger 20x10 nm
at d = 20 nm above the well, mu = 2e6 cm^2/Vs, tau0 = 12 ps at T0 = 4 K,
p = 1.5, t_transit = 0.24 ps.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrow

# Bit-identical figures across runs (see entangletron_experiment.py).
PDF_METADATA = {"CreationDate": None}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# Physical constants (SI)
E = 1.602176634e-19          # C
EPS0 = 8.8541878128e-12      # F/m
M0 = 9.1093837015e-31        # kg
HBAR = 1.054571817e-34       # J s
MSTAR = 0.042
MEV = 1e-3 * E               # J

# ---------------------------------------------------------------------------
# fig_poisson_mapping: gate voltage -> effective lens potential
# ---------------------------------------------------------------------------
def fig_poisson_mapping():
    """Gate voltage -> effective screened lens potential (real Poisson--TF).

    All curves come from :class:`electrostatics.PoissonTFLens` (Sec. 2.2):
    strip Laplace solution screened by the 2DEG via the Thomas--Fermi
    dielectric function 1/(1 + q_TF/|q|), amplitude-calibrated to
    50 meV/V.  The Gaussian widths sigma_x, sigma_y are fitted on the
    central well of the actual screened profile (not assumed).
    """
    from entangletronica.electrostatics import PoissonTFLens
    lens = PoissonTFLens()
    rep = lens.mapping_report()
    sx, sy = rep["sigma_x_nm"], rep["sigma_y_nm"]
    slope, r2 = rep["slope_meV_per_V"], rep["r2_shallow"]
    Vg_op = -0.3

    # 1D cuts of the real screened profile at the operating point V_g = -0.3 V
    xs = np.linspace(-40, 40, 401)
    Xg, Yg = np.meshgrid(xs, [0.0], indexing="ij")
    prof_x = lens.get_lens(Xg, Yg, Vg_op)[:, 0]
    Xg2, Yg2 = np.meshgrid([0.0], xs, indexing="ij")
    prof_y = lens.get_lens(Xg2, Yg2, Vg_op)[0, :]
    prof_x /= np.abs(prof_x).max()
    prof_y /= np.abs(prof_y).max()

    fig = plt.figure(figsize=(11, 3.6))
    ax = fig.add_subplot(131)
    ax.plot(xs, prof_x, "C0", lw=1.6,
            label=rf"along $x$ ($\sigma_x={sx:.1f}$ nm, fit)")
    ax.plot(xs, prof_y, "C1", lw=1.6,
            label=rf"along $y$ ($\sigma_y={sy:.1f}$ nm, fit)")
    ax.axvline(-sx, ls=":", color="C0", lw=0.8)
    ax.axvline(sx, ls=":", color="C0", lw=0.8)
    ax.axvline(-sy, ls=":", color="C1", lw=0.8)
    ax.axvline(sy, ls=":", color="C1", lw=0.8)
    ax.set_xlabel("lateral position [nm]")
    ax.set_ylabel(r"$V_{\mathrm{eff}}$ [norm.]")
    ax.set_title(r"(a) Screened lens profile at $V_g=-0.3$ V")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(132)
    Vg = np.linspace(-0.5, 0.0, 251)
    V0 = lens.lens_depth(Vg)
    ax.plot(Vg, V0, "o-", ms=3, color="C2", lw=1.2)
    ax.axvline(Vg_op, ls="--", color="0.5", lw=1.0)
    ax.axvspan(Vg_op, 0, color="C4", alpha=0.12)
    ax.plot([Vg_op], [lens.lens_depth(Vg_op)], "ks", ms=6)
    ax.annotate(rf"$V_0={lens.lens_depth(Vg_op):.0f}$ meV @ $V_g={Vg_op:.1f}$ V",
                xy=(Vg_op, lens.lens_depth(Vg_op)),
                xytext=(-0.47, lens.lens_depth(Vg_op) + 2.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"gate voltage $V_g$ [V]")
    ax.set_ylabel(r"lens depth $V_0$ [meV]")
    ax.set_title("(b) Voltage-to-potential mapping")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(133)
    shallow = np.abs(Vg) <= 0.3
    ax.plot(Vg, V0, "C2", lw=1.2, alpha=0.35)
    ax.plot(Vg[shallow], V0[shallow], "C3", lw=1.8,
            label=rf"linear fit: {slope:.0f} meV/V, $R^2={r2:.3f}$")
    ax.axvspan(Vg_op, 0, color="C4", alpha=0.12)
    ax.text(-0.15, 2.5, "shallow regime\n$|V_0|\\ll E_F$", fontsize=8,
            ha="center", color="C4")
    ax.set_xlabel(r"gate voltage $V_g$ [V]")
    ax.set_ylabel(r"$V_0$ [meV]")
    ax.set_title("(c) Linearity of the mapping")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Poisson--Thomas--Fermi electrostatics: gate voltage to effective lens potential"
                 rf"  ($q_{{\mathrm{{TF}}}}={rep['q_TF_nm-1']:.3f}$ nm$^{{-1}}$)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# fig_coherence: ensemble visibility vs temperature
# ---------------------------------------------------------------------------
def fig_coherence():
    """Ensemble visibility vs temperature: REAL numerical results.

    Loads results/coherence_ensemble.json (produced by
    scripts/ensemble_coherence.py, Sec. 2.4): N_ens = 200 realisations of the
    delta-correlated dephasing model per temperature, C_numerical = visibility
    of the ensemble-mean detector profile, C_std = bootstrap 1-sigma band.
    """
    import json
    path = os.path.join(HERE, "results", "coherence_ensemble.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run scripts/ensemble_coherence.py first "
            "(CI does this before make_missing_figures.py)")
    d = json.load(open(path))
    T_num = np.array(d["temperatures"])
    C_num = np.array(d["C_numerical"])
    C_std = np.array(d["C_std"])
    C_ana = np.array(d["C_analytical"])
    Tc = d.get("T_cross_half_K")

    tau0, T0, p = 12.0, 4.0, 1.5
    t_transit = 0.24
    C0 = 0.95
    T = np.logspace(np.log10(4), np.log10(77), 400)
    tau = tau0 * (T0 / T) ** p
    C = C0 * np.exp(-t_transit / tau)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(T, C, "C0", lw=1.8,
            label=r"$C_{\mathrm{ana}}(T)=0.95\,e^{-t_{\rm tr}/\tau_\phi(T)}$")
    ax.fill_between(T_num, C_num - C_std, C_num + C_std, color="C3", alpha=0.22,
                    label=rf"$1\sigma$ band ($N_\mathrm{{ens}}={d['N_ens']}$)")
    ax.errorbar(T_num, C_num, yerr=C_std, fmt="o", ms=5, capsize=3,
                color="C3", ecolor="C3", elinewidth=1.3,
                label="numerical ensemble")
    ax.axhline(0.5, ls="--", color="k", lw=0.9)
    ax.text(6.2, 0.52, "operating bound $C=0.5$", fontsize=8, color="0.3")
    if Tc is not None:
        ax.axvline(Tc, ls=":", color="C3", lw=1.1)
        ax.text(Tc * 1.18, 0.30, rf"$T_{{\rm max}}\approx{Tc:.0f}$ K",
                color="C3", fontsize=9)
    ax.axvspan(77, 80, color="C4", alpha=0.10)
    ax.text(79.3, 0.12, "LN$_2$: washed out", fontsize=8, color="C4", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(4, 80)
    ax.set_xlabel(r"temperature $T$ [K]")
    ax.set_ylabel(r"fringe visibility $C(T)$")
    ax.set_title(r"Coherence budget: $\mu=2\times10^6$ cm$^2$/Vs, $\tau_0=12$ ps at 4 K"
                 rf" (ensemble, $\mathrm{{scale_{{noise}}}}={d['scale_noise']:.0f}$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_xor_schematic: two-lens XOR proposal
# ---------------------------------------------------------------------------
def fig_xor_schematic():
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(121)
    ax.set_xlim(-10, 150); ax.set_ylim(-50, 55)
    ax.axis("off")
    # channel
    ax.plot([-5, 140], [34, 34], "k-", lw=2)
    ax.plot([-5, 140], [-34, -34], "k-", lw=2)
    ax.text(60, 41, "2DEG channel", ha="center", fontsize=8, color="0.3")
    # source
    src = Ellipse((6, 0), 10, 22, fc="#ffd0d0", ec="k")
    ax.add_patch(src)
    ax.text(6, 0, "e$^-$", ha="center", va="center", fontsize=10)
    # double-slit barrier
    ax.add_patch(Rectangle((48, -34), 6, 68, fc="#c0c0c0", ec="k"))
    ax.text(51, 40, "barrier", fontsize=8, ha="center", rotation=90)
    # two lenses behind the two slits
    la = Ellipse((66, 12), 12, 12, fc="#fff0b0", ec="k")
    lb = Ellipse((66, -12), 12, 12, fc="#ffd8a8", ec="k")
    ax.add_patch(la); ax.add_patch(lb)
    ax.text(73, 17, r"lens A: $V_1$", fontsize=9)
    ax.text(73, -19, r"lens B: $V_2$", fontsize=9)
    # detector
    ax.add_patch(Rectangle((120, -14), 4, 28, fc="#d0ffd0", ec="k"))
    ax.text(122, 24, "detector", fontsize=8, ha="center")
    ax.text(122, -40, r"$\mathcal{I}>\mathcal{I}_{\mathrm{th}}\Rightarrow$ logic 1", fontsize=8, ha="center", color="C2")
    ax.text(122, -47, r"$\mathcal{I}<\mathcal{I}_{\mathrm{th}}\Rightarrow$ logic 0", fontsize=8, ha="center", color="C3")
    # waves
    ax.annotate("", xy=(44, 8), xytext=(16, 6), arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.6))
    ax.annotate("", xy=(44, -8), xytext=(16, -6), arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.6))
    ax.text(30, -30, "coherent double-slit source", fontsize=8, color="C0", ha="center")
    ax.set_title("(a) Dual-lens interferometer")

    ax = fig.add_subplot(122)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.5, 1.1)
    rows = [
        ("0", "0", "balanced (reset)", "0 / 1", "grey"),
        ("0", "1", "lower lens on", "0", "C3"),
        ("1", "0", "upper lens on", "1", "C2"),
        ("1", "1", "balanced (reset)", "0 / 1", "grey"),
    ]
    for i, (v1, v2, cond, out, col) in enumerate(rows):
        y = 0.92 - i * 0.34
        ax.add_patch(Rectangle((-0.5, y - 0.13), 0.9, 0.26, fc="white", ec="k", lw=0.8))
        ax.text(0.0, y, v1, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.add_patch(Rectangle((0.45, y - 0.13), 0.9, 0.26, fc="white", ec="k", lw=0.8))
        ax.text(0.9, y, v2, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(1.5, y + 0.03, cond, fontsize=9, va="center")
        ax.text(2.28, y, out, ha="center", va="center", fontsize=11,
                fontweight="bold", color=col)
    ax.text(0.0, 1.04, r"$V_1$", ha="center", fontsize=9)
    ax.text(0.9, 1.04, r"$V_2$", ha="center", fontsize=9)
    ax.text(2.28, 1.04, "out", ha="center", fontsize=9)
    ax.text(-0.45, -0.38, r"$\Delta\phi=\phi(V_1)-\phi(V_2)$  linearises the XOR truth table "
                          r"(Eq. 13)", fontsize=8, color="0.25")
    ax.set_title("(b) XOR truth table (shallow-lens regime)")
    ax.axis("off")

    fig.suptitle("Scalable two-input XOR gate with independent electrostatic lenses",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


if __name__ == "__main__":
    figs = {
        "fig_poisson_mapping": fig_poisson_mapping(),
        "fig_coherence": fig_coherence(),
        "fig_xor_schematic": fig_xor_schematic(),
    }
    for name, f in figs.items():
        p = os.path.join(FIG, name + ".pdf")
        f.savefig(p, bbox_inches="tight", metadata=PDF_METADATA)
        f.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(f)
        print("wrote", p)
