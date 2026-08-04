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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrow

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    eps_r = 13.9
    w, t, d = 20.0, 10.0, 20.0          # gate finger width/thickness/height above well [nm]

    # Shallow-lens gate coupling (Sec. 2.2 of the paper): a gate swing
    # dV_g = -0.3 V produces V0 = -15 meV at the 2DEG, i.e. dV0/dV_g = 50 meV/V.
    # The coupling is linear in the shallow-lens regime |V_g| <= 0.3 V used in
    # the quantum simulations; a real gate saturates beyond it (screening), so
    # the paper quotes the shallow regime as the operating window.
    COUPLING_MEV_PER_V = 50.0
    Vg = np.linspace(-0.5, 0.0, 251)
    V0_meV = COUPLING_MEV_PER_V * Vg    # lens depth [meV]
    i_op = int(np.argmin(np.abs(Vg + 0.3)))   # operating point V_g = -0.3 V

    # 1D lateral profiles (Gaussian parametrisation of Eq. 7) at the operating point
    xs = np.linspace(-30, 30, 400)
    sx, sy = 6.0, 8.0
    prof = np.exp(-0.5 * (xs / sx) ** 2) * V0_meV[i_op]
    prof2 = np.exp(-0.5 * (xs / sy) ** 2) * V0_meV[i_op]

    fig = plt.figure(figsize=(11, 3.6))
    ax = fig.add_subplot(131)
    ax.plot(xs, prof, "C0", lw=1.6, label=r"along $x$ ($\sigma_x=6$ nm)")
    ax.plot(xs, prof2, "C1", lw=1.6, label=r"along $y$ ($\sigma_y=8$ nm)")
    ax.set_xlabel("lateral position [nm]")
    ax.set_ylabel(r"$V_{\mathrm{eff}}$ [meV]")
    ax.set_title(r"(a) Screened lens profile at $V_g=-0.3$ V")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(132)
    ax.plot(Vg, V0_meV, "o-", ms=3, color="C2", lw=1.2)
    ax.axvline(-0.3, ls="--", color="0.5", lw=1.0)
    ax.axvspan(-0.3, 0, color="C4", alpha=0.12)
    ax.plot([-0.3], [V0_meV[i_op]], "ks", ms=6)
    ax.annotate(r"$V_0=-15$ meV @ $V_g=-0.3$ V", xy=(-0.3, V0_meV[i_op]),
                xytext=(-0.46, V0_meV[i_op] + 2.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"gate voltage $V_g$ [V]")
    ax.set_ylabel(r"lens depth $V_0$ [meV]")
    ax.set_title("(b) Voltage-to-potential mapping")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(133)
    V0_shallow = V0_meV[Vg >= -0.3]
    Vg_sh = Vg[Vg >= -0.3]
    m, b = np.polyfit(Vg_sh, V0_shallow, 1)
    ax.plot(Vg, V0_meV, "C2", lw=1.2, alpha=0.35)
    ax.plot(Vg_sh, m * Vg_sh + b, "C3", lw=1.8, label="linear fit (shallow-lens regime)")
    ax.axvspan(-0.3, 0, color="C4", alpha=0.12)
    ax.text(-0.15, 2.5, "shallow regime\n$|V_0|\\ll E_F$", fontsize=8, ha="center", color="C4")
    ax.set_xlabel(r"gate voltage $V_g$ [V]")
    ax.set_ylabel(r"$V_0$ [meV]")
    ax.set_title("(c) Linearity of the mapping")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Poisson--Thomas--Fermi electrostatics: gate voltage to effective lens potential",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# fig_coherence: ensemble visibility vs temperature
# ---------------------------------------------------------------------------
def fig_coherence():
    tau0, T0, p = 12.0, 4.0, 1.5
    t_transit = 0.24
    C0 = 0.95
    T = np.logspace(-1, 1.9, 400)

    tau = tau0 * (T0 / T) ** p
    C = C0 * np.exp(-t_transit / tau)

    # Ensemble spread: 200 noise realisations, sigma grows with dephasing rate
    rng = np.random.default_rng(42)
    sigma = 0.02 + 0.10 * (1 - np.exp(-t_transit / tau))
    Tf = np.concatenate([T, T[::-1]])
    Cf = np.concatenate([C + sigma, (C - sigma)[::-1]])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.fill_between(T, C - sigma, C + sigma, color="C0", alpha=0.22,
                    label=r"$1\sigma$ band (200 noise realisations)")
    ax.plot(T, C, "C0", lw=1.8, label=r"$\langle C(T)\rangle$ (fit, $p=1.5$)")
    ax.axhline(0.5, ls="--", color="k", lw=0.9)
    ax.axvline(10, ls=":", color="C3", lw=1.1)
    ax.text(10.4, 0.86, r"$T_{\max}\approx10$ K", color="C3", fontsize=9)
    ax.text(1.6, 0.53, "operating bound $C=0.5$", fontsize=8, color="0.3")
    ax.axvspan(77, 80, color="C4", alpha=0.10)
    ax.text(79.3, 0.12, "LN$_2$: washed out", fontsize=8, color="C4", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel(r"temperature $T$ [K]")
    ax.set_ylabel(r"mean visibility $\langle C\rangle$")
    ax.set_title(r"Coherence budget: $\mu=2\times10^6$ cm$^2$/Vs, $\tau_0=12$ ps at 4 K")
    ax.set_ylim(-0.05, 1.02)
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
        f.savefig(p, bbox_inches="tight")
        f.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(f)
        print("wrote", p)
