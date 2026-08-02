"""Entangletron full experiment + figure pipeline.

Single run reproduces every figure of the paper:

    python scripts/entangletron_experiment.py

Outputs: figures/*.pdf|png  and  results/entangletron_metrics.json
"""

import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entangletronica import potential as P
from entangletronica import electron
from entangletronica import gates

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

# ------------------------------------------------------------------ grid & packet
NX, NY, DX = 140, 80, 2.0
X = np.arange(NX) * DX - 40.0
Y = np.arange(NY) * DX - 80.0
xx, yy = np.meshgrid(X, Y, indexing="ij")
K0, S = 0.2, 10.0
DT, NT = 0.30, 1300
psi0 = electron.gaussian_packet(xx, yy, k0=K0, s=S)

SLIT_Y = 12.0       # slit centres
SLIT_S = 4.0        # slit width
BARRIER_X = 60.0    # barrier position
PHASE_X = 68.0      # phase lens behind upper slit
DET_X = 110.0       # detector plane


def young_landscape(x, y, Vg=0.0, phase_k=0.0, barrier_k=1.0, barrier_a=12.0):
    """Double slit at y=+-SLIT_Y + electrostatic phase gate on the upper slit."""
    V = np.zeros_like(x)
    V += P.wall(x, y, xx=0.0, w=6.0, a=15.0)
    V += barrier_k * barrier_a * P.gauss(x, BARRIER_X, 6.0) * \
         (1.0 - P.gauss(y, -SLIT_Y, SLIT_S)) * (1.0 - P.gauss(y, SLIT_Y, SLIT_S))
    V += P.phase_shifter(x, y, x0=PHASE_X, y0=SLIT_Y, s=6.0, a=-15.0, k=phase_k)
    V += P.wall(x, y, xx=160.0, w=6.0, a=15.0)
    return V


def run(phase_k=0.0, barrier_a=12.0):
    return electron.run_landscape(young_landscape, X, Y, DT, NT, psi0=psi0,
                                  phase_k=phase_k, barrier_a=barrier_a)


def detector_imbalance(psi, xdet=DET_X, bin_half=14.0):
    """(P_upper - P_lower) in two bins at the detector plane, normalised."""
    p = np.abs(psi) ** 2
    i = int(np.argmin(np.abs(X - xdet)))
    prof = p[i, :]
    iu = (Y >= 0) & (Y < bin_half)
    il = (Y > -bin_half) & (Y < 0)
    pu, pl = prof[iu].sum(), prof[il].sum()
    tot = pu + pl
    return (pu - pl) / tot if tot > 0 else 0.0, pu, pl


def fringe_peak(psi, xdet=DET_X):
    p = np.abs(psi) ** 2
    i = int(np.argmin(np.abs(X - xdet)))
    prof = p[i, :]
    peaks, _ = find_peaks(prof, prominence=1e-5)
    if len(peaks) == 0:
        return np.nan
    return Y[peaks[np.argmax(prof[peaks])]]


# ------------------------------------------------------------------ EXPERIMENTS
def experiment_transfer():
    """Fine gate sweep: imbalance vs phase_k (the transfer characteristic)."""
    phis = np.linspace(0.0, 2.5, 16)
    out = []
    for pk in phis:
        psi, hist, norm, Vmev = run(phase_k=pk)
        imb, pu, pl = detector_imbalance(psi)
        out.append((pk, imb, pu, pl, norm))
    return np.array(out)


def experiment_visibility_vs_slits():
    """Peak position and contrast of the interference figure vs gate."""
    phis = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    out = []
    for pk in phis:
        psi, _, norm, _ = run(phase_k=pk)
        yp = fringe_peak(psi)
        p = np.abs(psi) ** 2
        i = int(np.argmin(np.abs(X - DET_X)))
        prof = p[i, :]
        vis = (prof.max() - prof.min()) / (prof.max() + prof.min() + 1e-12)
        out.append((pk, yp, vis, norm))
    return np.array(out)


def experiment_wavefunction(phase_k):
    psi, hist, norm, Vmev = run(phase_k=phase_k)
    return psi, Vmev, norm


# ------------------------------------------------------------------ FIGURES
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(-40, 160); ax.set_ylim(-50, 50)
    ax.axis("off")
    ax.set_title("Entangletronic chip: flying electron + electrostatic lens gate")
    # channel
    ax.plot([-30, 150], [34, 34], 'k-', lw=2)
    ax.plot([-30, 150], [-34, -34], 'k-', lw=2)
    ax.text(60, 40, "2DEG channel", ha="center", fontsize=9, color="0.3")
    # source
    from matplotlib.patches import Ellipse
    src = Ellipse((-18, 0), 10, 22, fc="#ffd0d0", ec="k")
    ax.add_patch(src)
    ax.text(-18, 0, "e$^-$", ha="center", va="center", fontsize=11)
    # barrier with two slits
    ax.add_patch(plt.Rectangle((BARRIER_X - 3, -34), 6, 68, fc="#c0c0c0", ec="k"))
    ax.text(BARRIER_X - 2, 40, "double slit", fontsize=8, rotation=90, va="top")
    # phase gate on upper slit
    gate = Ellipse((PHASE_X, SLIT_Y), 10, 10, fc="#fff0b0", ec="k")
    ax.add_patch(gate)
    ax.text(PHASE_X + 8, SLIT_Y + 2, r"$V_\phi$ (gate)", fontsize=9)
    # detector
    ax.add_patch(plt.Rectangle((DET_X - 1.5, -14), 3, 28, fc="#d0ffd0", ec="k"))
    ax.text(DET_X, 24, "detector", fontsize=9, ha="center")
    # wave arrow
    ax.annotate("", xy=(BARRIER_X - 10, 0), xytext=(-8, 0),
                arrowprops=dict(arrowstyle="-|>", color="C0", lw=2))
    ax.text(15, 8, "flying wave packet", fontsize=9, color="C0")
    ax.text(110, -40, "interference figure (fringes)", fontsize=8, color="C3")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def fig_free():
    # free flight only (guide walls, no slits, no gate)
    psi, _, _, _ = electron.run_landscape(
        lambda x, y, **kw: P.wall(x, y, xx=160.0, w=6.0, a=15.0) + P.wall(x, y, xx=0.0, w=6.0, a=15.0),
        X, Y, DT, NT, psi0=psi0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, psi_s, tt in zip(axes, [psi0, psi], ["t = 0", "after free flight"]):
        im = ax.imshow(np.abs(psi_s.T) ** 2, extent=(X[0], X[-1], Y[0], Y[-1]),
                       origin="lower", cmap="magma", aspect="auto")
        ax.set_title(tt)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    fig.suptitle("Flying electron: free propagation of the probability wave")
    fig.tight_layout()
    return fig


def fig_interference():
    """Final |psi|^2 at the detector plane + gate ON/OFF fringe shift."""
    psi_off, _, _, Vmev_off = run(phase_k=0.0)
    psi_on, _, _, Vmev_on = run(phase_k=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    im = ax.imshow(np.abs(psi_on.T) ** 2, extent=(X[0], X[-1], Y[0], Y[-1]),
                   origin="lower", cmap="magma", aspect="auto")
    ax.contour(X, Y, Vmev_on.T, levels=[5, 10], colors="w", linewidths=0.6, alpha=0.5)
    ax.set_title(r"Interference figure, gate ON ($V_\phi$ on upper slit)")
    ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")

    ax = axes[1]
    i = int(np.argmin(np.abs(X - DET_X)))
    for psi_s, c, lab in [(psi_off, "C0", "gate OFF"), (psi_on, "C3", "gate ON")]:
        prof = np.abs(psi_s) ** 2
        prof = prof[i, :] / prof[i, :].max()
        ax.plot(Y, prof, c, lw=1.6, label=lab)
    ax.set_xlabel("y [nm]")
    ax.set_ylabel(r"$|\psi|^2$ (normalised)")
    ax.set_title("Detector-plane profile: fringe shift by the gate")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Electrostatic phase gate shifts the interference figure")
    fig.tight_layout()
    return fig


def fig_transfer(data):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(data[:, 0], data[:, 1], "o-", color="C2", lw=1.8)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("gate depth factor $k_\\phi$  (0 = off, 1 = $-15$ meV)")
    ax.set_ylabel("detector imbalance  $(P_U-P_L)/(P_U+P_L)$")
    ax.set_title("Transfer characteristic: gate-controlled interference logic")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_peak(data):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(data[:, 0], data[:, 1], "s-", color="C3", lw=1.8, label="fringe peak $y_p$")
    ax.plot(data[:, 0], data[:, 2], "o-", color="C1", lw=1.8, label="profile contrast")
    ax.set_xlabel("gate depth factor $k_\\phi$")
    ax.set_ylabel("peak position [nm] / contrast")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Fringe displacement and contrast vs gate")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ MAIN
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()

    print("[1/5] transfer characteristic (16 runs)...")
    tr = experiment_transfer()
    np.save(os.path.join(RES, "transfer.npy"), tr)

    print("[2/5] fringe peak & contrast (6 runs)...")
    pk = experiment_visibility_vs_slits()
    np.save(os.path.join(RES, "fringe.npy"), pk)

    print("[3/5] figures...")
    figs = {
        "fig1_architecture": fig_architecture(),
        "fig2_free_propagation": fig_free(),
        "fig3_interference": fig_interference(),
        "fig4_transfer": fig_transfer(tr),
        "fig5_fringe": fig_peak(pk),
    }
    for name, fig in figs.items():
        fig.savefig(os.path.join(FIG, name + ".pdf"))
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=150)
        plt.close(fig)
        print("   wrote", name)

    print("[4/5] metrics...")
    # sensitivity & linearity
    k = tr[:, 0]
    imb = tr[:, 1]
    m, b = np.polyfit(k, imb, 1)
    resid = imb - (m * k + b)
    r2 = 1 - resid.var() / imb.var()
    imax = np.abs(imb).max()
    vis_max = np.abs((tr[:, 2] - tr[:, 3]) / (tr[:, 2] + tr[:, 3] + 1e-12)).max()
    metrics = {
        "energy_meV": 0.5 * K0 ** 2 / P.MEV_TO_NAT,
        "deBroglie_nm": 2 * np.pi / K0,
        # Young two-slit fringe spacing at the detector plane: d_l = (L*lambda)/a
        "fringe_period_est_nm": (DET_X - BARRIER_X) * (2 * np.pi / K0) / (2 * SLIT_Y),
        "transfer_slope_per_kphi": float(m),
        "transfer_linear_r2": float(r2),
        "max_imbalance": float(imax),
        "max_bin_visibility": float(vis_max),
        "norm_conservation_max_dev": float(np.max(np.abs(tr[:, 4] - 1.0))),
        "detector_plane_x_nm": DET_X,
    }
    # attach the fixed-time convergence study if it has been run
    conv_path = os.path.join(RES, "convergence.json")
    if os.path.exists(conv_path):
        conv = json.load(open(conv_path))
        metrics["convergence_imbalance_by_dx"] = [
            {"dx_nm": c["dx"], "dt": c["dt"], "imbalance": c["imbalance"]}
            for c in conv
        ]
    with open(os.path.join(RES, "entangletron_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("   metrics:", json.dumps(metrics, indent=2))

    print(f"[5/5] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
