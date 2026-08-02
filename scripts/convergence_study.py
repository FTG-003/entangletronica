"""Numerical grid-refinement study (the meaningful validation the R2 metric is not).

Examines spatial convergence of the interference observable (detector imbalance,
fringe peak position and band transmission at phase_k=1.0) across grid
refinements:

    Delta x : {4, 2, 1, 0.5} nm
    Delta t : scaled to keep a FIXED physical propagation time (TSIM units)

Fixing the physical time (not the step count) is essential: otherwise each
case travels a different distance and the curves are not comparable. A
converged result means the observable settles as the grid is refined --- the
reported value is a property of the Schroedinger dynamics, not of the lattice.

Run: python scripts/convergence_study.py  (~ tens of seconds)
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entangletronica import potential as P
from entangletronica import electron

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

# physical device parameters (match entangletron_experiment.py)
SLIT_Y, BARRIER_X, DET_X = 12.0, 60.0, 110.0
K0, S = 0.2, 10.0
TSIM = 390.0  # fixed physical propagation time (solver units)


def young(x, y, phase_k=1.0, barrier_a=12.0):
    """Same landscape as entangletron_experiment.py."""
    V = np.zeros_like(x)
    V += P.wall(x, y, xx=0.0, w=6.0, a=15.0)
    V += barrier_a * P.gauss(x, BARRIER_X, 6.0) * \
         (1.0 - P.gauss(y, -SLIT_Y, 4.0)) * (1.0 - P.gauss(y, SLIT_Y, 4.0))
    V += P.phase_shifter(x, y, x0=68.0, y0=SLIT_Y, s=6.0, a=-15.0, k=phase_k)
    V += P.wall(x, y, xx=160.0, w=6.0, a=15.0)
    return V


def observable(psi, X, Y):
    """Detector-plane observables at x=DET_X.

    - imbalance   (Pu-Pl)/(Pu+Pl), 2-bin read-out in |y| <= 14 nm
    - fringe      y-position of the probability maximum within the band
    - transmission total band probability |y| <= 14 nm
    """
    p = np.abs(psi) ** 2
    i = int(np.argmin(np.abs(X - DET_X)))
    prof = p[i, :]
    band = (Y >= -14.0) & (Y <= 14.0)
    iu = (Y >= 0.0) & (Y <= 14.0)
    il = (Y >= -14.0) & (Y < 0)
    pu, pl = prof[iu].sum(), prof[il].sum()
    imb = (pu - pl) / (pu + pl) if pu + pl > 0 else 0.0
    # sub-grid fringe peak via parabolic fit about the discrete max
    yb = Y[band]
    pb = prof[band]
    j = int(np.argmax(pb))
    y0 = yb[j]
    if 0 < j < len(yb) - 1:
        dy = yb[1] - yb[0]
        a, b2, c = pb[j - 1], pb[j], pb[j + 1]
        den = a - 2 * b2 + c
        if abs(den) > 1e-15:
            y0 += 0.5 * dy * (a - c) / den
    trans = float(pb.sum())
    return imb, y0, trans


def main():
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()

    # dx decreases with dt so that dx/dt (Courant) stays fixed -> same physical time.
    grid_cases = [
        {"dx": 4.0,  "Nx": 110, "Ny": 60,  "dt": 0.60, "Nt": int(TSIM / 0.60)},
        {"dx": 2.0,  "Nx": 140, "Ny": 80,  "dt": 0.30, "Nt": int(TSIM / 0.30)},
        {"dx": 1.0,  "Nx": 260, "Ny": 150, "dt": 0.15, "Nt": int(TSIM / 0.15)},
        {"dx": 0.5,  "Nx": 520, "Ny": 300, "dt": 0.075, "Nt": int(TSIM / 0.075)},
    ]

    rows = []
    for c in grid_cases:
        X = np.arange(c["Nx"]) * c["dx"] - 40.0
        Y = np.arange(c["Ny"]) * c["dx"] - 64.0
        xx, yy = np.meshgrid(X, Y, indexing="ij")
        psi0 = electron.gaussian_packet(xx, yy, k0=K0, s=S)
        Vmev = young(xx, yy)
        V = Vmev * P.MEV_TO_NAT
        psi = psi0.copy()
        for _ in range(c["Nt"]):
            psi = electron.step(psi, V, c["dt"], X, Y)
        imb, fringe, trans = observable(psi, X, Y)
        norm = float(np.sum(np.abs(psi) ** 2))
        rows.append({"dx": c["dx"], "dt": c["dt"], "Nt": c["Nt"],
                     "imbalance": imb, "fringe_peak_nm": fringe,
                     "band_transmission": trans, "norm": norm})
        print(f"  dx={c['dx']:g} dt={c['dt']:.3g}: imbalance={imb:.4f} "
              f"fringe={fringe:+.2f}nm trans={trans:.3f} norm={norm:.6f}")

    # relative change of imbalance between successive refinements, and whether
    # the last step is below a 1% threshold (the user-specified convergence gate)
    crit = []
    for a, b in zip(rows[:-1], rows[1:]):
        rel = abs((b["imbalance"] - a["imbalance"]) / b["imbalance"]) if b["imbalance"] else float("nan")
        crit.append({"from_dx": a["dx"], "to_dx": b["dx"],
                     "imbalance": b["imbalance"], "rel_change": rel})
    rows_sorted = rows  # already descending dx
    report = {
        "grid_cases": rows_sorted,
        "relative_change": crit,
        "imbalance_converged_under_1pct": bool(
            abs(crit[-1]["rel_change"]) < 0.01) if len(crit) >= 2 and crit[-1]["rel_change"] == crit[-1]["rel_change"] else False,
        "observable": "detector imbalance at phase_k=1.0, fixed physical time 390 units",
    }

    json.dump(report, open(os.path.join(RES, "convergence.json"), "w"), indent=2)
    print(f"[convergence] wrote results/convergence.json  ({time.time()-t0:.0f}s)")
    print(f"[convergence] |dI| between last two refinements: "
          f"{crit[-1]['rel_change']*100:.2f}%  -> converged(1%): "
          f"{report['imbalance_converged_under_1pct']}")

if __name__ == "__main__":
    main()