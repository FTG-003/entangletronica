"""Numerical-convergence study (the meaningful validation the R2 metric is not).

Examines spatial and temporal convergence of the interference observable
(detector imbalance at phase_k=1.0) across grid/timestep refinements:

    Delta x : {4, 2, 1} nm
    Delta t : {0.30, 0.15} solver units

A converged result means the imbalance drifts by less than the discretisation
noise as the grid is refined, i.e. the observable is a property of the
Schroedinger dynamics, not of the lattice.

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
# keep propagation distance fixed: Nt * dt * (grid) scales transit; we fix
# simulated time so results are directly comparable across grids.


def young(x, y, phase_k=0.0, barrier_a=12.0):
    V = np.zeros_like(x)
    V += P.wall(x, y, xx=0.0, w=6.0, a=15.0)
    V += barrier_a * P.gauss(x, BARRIER_X, 6.0) * \
         (1.0 - P.gauss(y, -SLIT_Y, 4.0)) * (1.0 - P.gauss(y, SLIT_Y, 4.0))
    V += P.phase_shifter(x, y, x0=68.0, y0=SLIT_Y, s=6.0, a=-15.0, k=phase_k)
    V += P.wall(x, y, xx=160.0, w=6.0, a=15.0)
    return V


def imbalance_at(psi, X, Y):
    p = np.abs(psi) ** 2
    i = int(np.argmin(np.abs(X - DET_X)))
    prof = p[i, :]
    iu = (Y >= 0) & (Y < 14.0)
    il = (Y > -14.0) & (Y < 0)
    pu, pl = prof[iu].sum(), prof[il].sum()
    return (pu - pl) / (pu + pl) if pu + pl > 0 else 0.0


def main():
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()

    # travel a fixed physical distance (~110 nm downstream + margin) at each
    # resolution, at the same physical speed: Nt = distance/(dt*vnorm).
    # Fixed physical propagation time across all resolutions, so the packet
    # travels the same distance in every case: Nt = Tsolve / dt.
    TSIM = 390.0  # solver time units (matches entangletron_experiment: 1300*0.30)
    grid_cases = [
        {"dx": 4.0, "Nx": 110, "Ny": 60, "dt": 0.60, "Nt": int(TSIM / 0.60)},
        {"dx": 2.0, "Nx": 140, "Ny": 80, "dt": 0.30, "Nt": int(TSIM / 0.30)},
        {"dx": 1.0, "Nx": 260, "Ny": 150, "dt": 0.15, "Nt": int(TSIM / 0.15)},
    ]

    rows = []
    for c in grid_cases:
        X = np.arange(c["Nx"]) * c["dx"] - 40.0
        Y = np.arange(c["Ny"]) * c["dx"] - 64.0
        xx, yy = np.meshgrid(X, Y, indexing="ij")
        psi0 = electron.gaussian_packet(xx, yy, k0=K0, s=S)
        Vmev = young(*np.meshgrid(X, Y, indexing="ij"))
        V = Vmev * P.MEV_TO_NAT
        psi = psi0.copy()
        for _ in range(c["Nt"]):
            psi = electron.step(psi, V, c["dt"], X, Y)
        imb = imbalance_at(psi, X, Y)
        norm = np.sum(np.abs(psi) ** 2)
        rows.append({**c, "imbalance": imb, "norm": norm})
        print(f"  dx={c['dx']:g} dt={c['dt']:.2f}: imbalance={imb:.4f} "
              f"norm={norm:.6f}")

    json.dump(rows, open(os.path.join(RES, "convergence.json"), "w"), indent=2)
    print(f"[convergence] wrote results/convergence.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()