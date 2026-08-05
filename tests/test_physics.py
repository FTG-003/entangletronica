"""
Physics-validation regression tests for the corrections spec (2025):

  1. Poisson--Thomas--Fermi electrostatics (electrostatics.py, paper Sec. 2.2):
     V_g = -0.3 V  ->  V0 = -15 meV, linear coupling 50 meV/V in the shallow
     regime, verified on the real screened potential.
  2. Stochastic dephasing solver (stochastic.py, paper Sec. 2.4): the
     delta-correlated noise must produce a genuine phase variance on top of the
     unitary deterministic evolution.
  3. Ensemble coherence vs temperature (scripts/ensemble_coherence.py):
     C(T) strictly monotonic-decreasing with T.
  4. 4 K calibration point: C_num(4 K) matches the analytical
     0.95*exp(-t_tr/tau_phi) value 0.931 (tolerance 0.05, ensemble fluctuation).

Tests 3-4 read the *committed* results/coherence_ensemble.json (N_ens = 200).
If it is absent (fresh clone) they fall back to a small fixed-seed ensemble so
the suite still exercises the machinery; CI generates the full JSON first.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from entangletronica import electrostatics, potential as P, electron, stochastic

ENSEMBLE_JSON = os.path.join(ROOT, "results", "coherence_ensemble.json")
TAU0_PS, T0_K, P_EXP, T_TRANSIT_PS, C0 = 12.0, 4.0, 1.5, 0.24, 0.95


def analytical_contrast(T):
    tau = TAU0_PS * (T0_K / T) ** P_EXP
    return C0 * np.exp(-T_TRANSIT_PS / tau)


# ---------------------------------------------------------------------------
# 1. Poisson--Thomas--Fermi lens
# ---------------------------------------------------------------------------
def test_poisson_tf_mapping():
    lens = electrostatics.PoissonTFLens()
    # operating point Vg = -0.3 V -> V0 = -15 meV
    deep = lens.lens_depth(-0.3)
    assert abs(deep - (-15.0)) <= 1.0, f"V0(-0.3V) = {deep} meV, expected ~-15"
    # centre of the screened potential reproduces the calibration
    centre = float(np.asarray(lens.get_lens(
        np.array([0.0]), np.array([0.0]), -0.3)).reshape(-1)[0])
    assert abs(centre - (-15.0)) <= 1.0, f"V_eff(0,0) = {centre} meV, expected ~-15"
    # shallow-regime slope = 50 meV/V from the real screening
    rep = lens.mapping_report()
    slope = rep["slope_meV_per_V"]
    assert 45.0 <= slope <= 55.0, f"slope = {slope} meV/V, expected ~50"
    assert rep["r2_shallow"] > 0.995
    # the Poisson--TF widths should be the physical ones, wider than the old toy
    assert 10.0 < rep["sigma_x_nm"] < 16.0
    assert 10.0 < rep["sigma_y_nm"] < 18.0


# ---------------------------------------------------------------------------
# 2. Stochastic dephasing
# ---------------------------------------------------------------------------
def _small_grid():
    NX, NY, DX = 48, 36, 2.0
    X = np.arange(NX) * DX - 24.0
    Y = np.arange(NY) * DX - 36.0
    return X, Y


def test_stochastic_dephasing_exists():
    """Adding delta-correlated noise must produce a nonzero phase variance
    relative to the deterministic (noise-free) propagation, while staying
    norm-conserving (split-step is unitary)."""
    X, Y = _small_grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=8.0)
    V = np.zeros_like(xx)
    dt, Nt = 0.1, 40
    tau_nat = 20.0 * P.PS_TO_NAT          # short dephasing time -> strong noise

    psi_det, _, _ = electron.solve2d(V, psi0, dt, Nt, X, Y, report=False)
    rng = np.random.default_rng(20260201)
    psi_noisy, norm = stochastic.solve2d_stochastic(
        V, psi0, dt, Nt, X, Y, tau_nat, rng, scale_noise=22.0)

    assert abs(norm - 1.0) < 1e-9, "noise must not break unitarity"
    ratio = psi_noisy / np.where(np.abs(psi_det) > 1e-12, psi_det, 1.0)
    phase_var = np.var(np.angle(ratio))
    assert phase_var > 1e-6, f"noise produced no phase variance ({phase_var:g})"
    # and it differs from the deterministic state (it must not be a copy)
    dev = np.max(np.abs(psi_noisy - psi_det))
    assert dev > 1e-6, "noisy state is identical to deterministic"


# ---------------------------------------------------------------------------
# 3 + 4. Ensemble coherence
# ---------------------------------------------------------------------------
def _load_ensemble():
    if os.path.exists(ENSEMBLE_JSON):
        return json.load(open(ENSEMBLE_JSON))
    return None


def _fallback_ensemble():
    """Tiny fixed-seed ensemble used only when the committed JSON is absent."""
    NX, NY, DX = 96, 60, 2.0
    X = np.arange(NX) * DX - 40.0
    Y = np.arange(NY) * DX - 64.0
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=10.0)
    dt, Nt = 0.30, 600
    V = P.young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT
    idet = int(np.argmin(np.abs(X - 110.0)))
    rng = np.random.default_rng(12345)
    T_list = [4.0, 10.0, 20.0, 50.0, 77.0]
    C = []
    for T in T_list:
        tau_nat = (TAU0_PS * (T0_K / T) ** P_EXP) * P.PS_TO_NAT
        acc = np.zeros(NY)
        for _ in range(6):
            psi, _ = stochastic.solve2d_stochastic(
                V, psi0, dt, Nt, X, Y, tau_nat, rng, scale_noise=16.0)
            pr = np.abs(psi) ** 2
            acc += pr[idet, :] / pr[idet, :].sum()
        acc /= 6.0
        C.append(float((acc.max() - acc.min()) / (acc.max() + acc.min())))
    return T_list, C


def test_ensemble_coherence_monotonic():
    """C(T) must be strictly decreasing with T (coherence is lost as T grows).
    Prefers the committed 200-realisation JSON; else a small fixed-seed run."""
    d = _load_ensemble()
    if d is not None:
        T = d["temperatures"]
        assert T == [4.0, 10.0, 20.0, 50.0, 77.0], f"unexpected T grid {T}"
        C = d["C_numerical"]
    else:
        T, C = _fallback_ensemble()
    for i in range(len(C) - 1):
        assert C[i] > C[i + 1], (
            f"C not strictly decreasing at index {i}: {C[i]:.3f} then {C[i+1]:.3f}")
    assert C[0] > 0.8, "1 K visibility implausibly low"


def test_ensemble_4K_matches_analytical():
    """Calibration anchor: numerical C(4 K) coincides with the analytical
    0.95*exp(-t_transit/tau_phi(4 K)) = 0.931 within 0.05."""
    d = _load_ensemble()
    if d is not None:
        C4 = d["C_numerical"][0]
        note = "from committed N_ens=200 JSON"
    else:
        _, C = _fallback_ensemble()
        C4 = C[0]
        note = "from small fallback ensemble (committed JSON absent)"
    target = analytical_contrast(4.0)
    assert abs(C4 - target) <= 0.05, (
        f"C(4K) = {C4:.3f} vs analytical {target:.3f} ({note})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed.")