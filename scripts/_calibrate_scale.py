"""Calibration sweep for scale_noise in the stochastic dephasing ensemble.

Probes C(T) = (max-min)/(max+min) of the ensemble-mean detector profile over the
full detector line, for candidate scale_noise values, and reports monotonicity
plus the 4 K match against the analytical curve C_ana(4K) = 0.95*exp(-t/ta).

Usage:  python3 scripts/_calibrate_scale.py [scale1 scale2 ...]
"""
import sys, time, numpy as np
from multiprocessing import Pool
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entangletronica import potential as P, electron, stochastic

NX, NY, DX = 140, 80, 2.0
X = np.arange(NX) * DX - 40.0
Y = np.arange(NY) * DX - 80.0
xx, yy = np.meshgrid(X, Y, indexing="ij")
psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=10.0)
DT, NT = 0.30, 1300
V = P.young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT
IDET = int(np.argmin(np.abs(X - 110.0)))
T_LIST = [4.0, 10.0, 20.0, 50.0, 77.0]
N_PROBE = 24


def contrast(prof):
    mx, mn = prof.max(), prof.min()
    return (mx - mn) / (mx + mn) if mx + mn > 0 else 0.0


def _job(args):
    T, scale, r = args
    tau_nat = (12.0 * (4.0 / T) ** 1.5) * P.PS_TO_NAT
    rng = np.random.default_rng(30000 + int(T * 100) + r)
    psi, _ = stochastic.solve2d_stochastic(V, psi0, DT, NT, X, Y, tau_nat, rng,
                                           scale_noise=scale)
    pr = np.abs(psi) ** 2
    return pr[IDET, :] / pr[IDET, :].sum()


def run_curve(scale, n=N_PROBE):
    tasks = [(T, scale, r) for T in T_LIST for r in range(n)]
    with Pool(4) as pool:
        results = pool.map(_job, tasks)
    C = {}
    for k, T in enumerate(T_LIST):
        acc = np.mean([results[k * n + r] for r in range(n)], axis=0)
        C[T] = contrast(acc)
    return C


if __name__ == "__main__":
    scales = [float(s) for s in sys.argv[1:]] or [22.0, 28.0, 35.0, 50.0]
    C_ana = {T: 0.95 * np.exp(-0.24 / (12.0 * (4.0 / T) ** 1.5)) for T in T_LIST}
    print("C_ana:", {T: round(C_ana[T], 3) for T in T_LIST}, flush=True)
    for scale in scales:
        t0 = time.time()
        C = run_curve(scale)
        seq = [C[t] for t in T_LIST]
        mono = all(seq[k] > seq[k + 1] for k in range(len(seq) - 1))
        print(f"scale={scale:4.0f}: C={[f'{v:.3f}' for v in seq]}  "
              f"C4K={C[4.0]:.3f} (target {C_ana[4.0]:.3f})  monotonic={mono}  "
              f"({time.time()-t0:.0f}s)", flush=True)
