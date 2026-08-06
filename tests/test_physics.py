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

Scientific-due-diligence additions (2026, referee-driven):
  5. Noiseless limit: scale_noise=0 must reproduce the deterministic solver
     exactly and C -> C_det ~ 1.0 (full-line convention); C_ana -> C0 = 0.95.
  6. Ensemble-size convergence (results/ensemble_convergence.json): C(4 K)
     converged to +-0.01 vs the operating N=200; the 10 K point honestly
     reports its larger band.
  7. Scale-factor sensitivity (results/scale_sensitivity.json): T_max strictly
     decreasing with the calibration scale s = 17/22/27.
  8. Spatially correlated noise (results/noise_correlation.json): at fixed
     local amplitude C(xi>0) >= C(xi=0) and T_max rises with xi (white noise
     is the conservative worst case).

v2 corrections-kit additions (2026):
  9. Differential selectivity (results/selectivity.json, Sec. 2.3): the
     cross-arm coupling ratio |V_eff(lower slit)/V_eff(upper slit)| must lie
     in [0.25, 0.40] (i.e. ~31% at the operating point) and the differential
     selectivity 1 - ratio in [0.60, 0.75]: the lens is NOT single-slit
     selective, and the paper says so.
 10. Dephasing calibration documented (results/coherence_ensemble.json): the
     committed JSON must carry a "calibration_note" stating that s_phi = 22
     is a consistency fix (the raw spec value 2.32 yields no dephasing), not
     a physical prediction.
 11. Text-honesty regression (paper/*.tex): the phrase "room temperature"
     must NOT appear anywhere in the manuscript, and "conservative
     white-noise bound" must appear at least once.

Tests 3-8 read the *committed* results/*.json (N_ens = 200).  If they are
absent (fresh clone) they fall back to a small fixed-seed ensemble so the
suite still exercises the machinery; CI regenerates the full JSONs first.
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
def _load_ensemble(name="coherence_ensemble.json"):
    path = os.path.join(ROOT, "results", name)
    if os.path.exists(path):
        return json.load(open(path))
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


# ---------------------------------------------------------------------------
# 5. Noiseless limit (scientific due diligence, referee point)
# ---------------------------------------------------------------------------
def test_noiseless_limit():
    """For tau_phi -> inf (zero noise) C must tend to the deterministic
    value.  Numerically the stochastic solver with scale_noise = 0 must reduce
    EXACTLY to the deterministic split-step solver, and the full-line contrast
    of that profile is the deterministic limit C_det ~ 1.0 (tails vanish).  The
    analytical envelope retains its single-particle prefactor C0 = 0.95, so
    both limits are checked."""
    X, Y = _small_grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=8.0)
    V = np.zeros_like(xx)
    dt, Nt = 0.1, 40
    psi_det, _, _ = electron.solve2d(V, psi0, dt, Nt, X, Y, report=False)
    rng = np.random.default_rng(20260201)
    psi_noiseless, norm = stochastic.solve2d_stochastic(
        V, psi0, dt, Nt, X, Y, 1.0, rng, scale_noise=0.0)
    assert abs(norm - 1.0) < 1e-9, "zero-noise run must stay unitary"
    assert np.max(np.abs(psi_noiseless - psi_det)) < 1e-12, (
        "scale_noise=0 must reproduce the deterministic evolution exactly")
    prof = np.abs(psi_det) ** 2
    prof = prof / prof.sum()
    C_det = (prof.max() - prof.min()) / (prof.max() + prof.min())
    assert C_det > 0.99, f"deterministic limit C_det = {C_det} (full-line ~1.0)"
    # analytical envelope in the infinite-tau limit -> C0 = 0.95
    assert abs(analytical_contrast(1e-6) - 0.95) < 1e-9, (
        "C_ana(tau->inf) must return the envelope prefactor C0 = 0.95")


# ---------------------------------------------------------------------------
# 6. Ensemble-size convergence (nested; committed JSON or small fallback)
# ---------------------------------------------------------------------------
def _fallback_convergence():
    """Tiny fixed-seed nested run on a small grid (committed JSON absent)."""
    X, Y = _small_grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=8.0)
    dt, Nt = 0.1, 400
    V = P.young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT
    tau_nat = (TAU0_PS * (T0_K / 4.0) ** P_EXP) * P.PS_TO_NAT
    rng = np.random.default_rng(9876)
    Nmax, idet = 12, int(np.argmin(np.abs(X - 60.0)))
    profs = []
    for _ in range(Nmax):
        psi, _ = stochastic.solve2d_stochastic(
            V, psi0, dt, Nt, X, Y, tau_nat, rng, scale_noise=22.0)
        p = np.abs(psi) ** 2
        profs.append(p[idet, :] / p[idet, :].sum())
    profs = np.array(profs)
    Ns, Cs = [3, 6, 12], []
    for n in Ns:
        m = profs[:n].mean(axis=0)
        Cs.append((m.max() - m.min()) / (m.max() + m.min()))
    return Ns, Cs


def test_ensemble_convergence():
    """C(4 K) must be converged with the ensemble size: |C(N)-C(200)| <= 0.01
    for every N >= 50 (nested analysis, deterministic seeds).  The 10 K point
    is the steepest (most ensemble-sensitive) and reports a larger honest band.
    Prefers results/ensemble_convergence.json; else a tiny fallback run."""
    d = _load_ensemble("ensemble_convergence.json")
    if d is not None:
        assert d["reference_operating_N"] == 200
        conv4 = d["convergence"]["4.0"]
        conv10 = d["convergence"]["10.0"]
        assert conv4["met"], (
            f"4 K not converged to {conv4['requested_tolerance']}: "
            f"max|dC| = {conv4['max_deviation_from_operating_N']}")
        # nested N=200 must reproduce the committed paper point (same seeds)
        C4_200 = d["C_nested"]["4.0"][d["n_ensemble_points"].index(200)]
        committed = _load_ensemble()["C_numerical"][0]
        assert abs(C4_200 - committed) < 0.005, (
            f"nested C(200,4K) = {C4_200:.4f} vs committed {committed:.4f}")
        assert conv10["max_deviation_from_operating_N"] >= conv4[
            "max_deviation_from_operating_N"], \
            "10 K must report the larger (honest) band"
    else:
        Ns, Cs = _fallback_convergence()
        assert abs(Cs[-1] - Cs[0]) < 0.2, \
            f"fallback C not converging: {dict(zip(Ns, Cs))}"


# ---------------------------------------------------------------------------
# 7. Scale-factor sensitivity (s = 22 +/- 5)
# ---------------------------------------------------------------------------
def _fallback_scale_sensitivity():
    """Direction check on a small grid: C(s=17) > C(s=27) at fixed T."""
    X, Y = _small_grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=8.0)
    dt, Nt = 0.1, 400
    V = P.young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT
    tau_nat = (TAU0_PS * (T0_K / 4.0) ** P_EXP) * P.PS_TO_NAT
    idet = int(np.argmin(np.abs(X - 60.0)))
    Cs = {}
    for s in [17.0, 27.0]:
        rng = np.random.default_rng(54321)
        acc = np.zeros(Y.size)
        for _ in range(10):
            psi, _ = stochastic.solve2d_stochastic(
                V, psi0, dt, Nt, X, Y, tau_nat, rng, scale_noise=s)
            p = np.abs(psi) ** 2
            acc += p[idet, :] / p[idet, :].sum()
        acc /= 10.0
        Cs[s] = (acc.max() - acc.min()) / (acc.max() + acc.min())
    return Cs


def test_scale_sensitivity():
    """T_max must decrease as the noise scale grows: T_max(17) > T_max(22) >
    T_max(27), with dT_max/ds < 0.  Prefers results/scale_sensitivity.json
    (N=200 scan); else a tiny fallback direction check."""
    d = _load_ensemble("scale_sensitivity.json")
    if d is not None:
        tm = d["Tmax_by_scale_K"]
        assert tm["17.0"] > tm["22.0"] > tm["27.0"], \
            f"T_max must decrease with s: {tm}"
        assert d["dTmax_ds_K_per_unit_s"] < 0
        assert tm["22.0"] == d["Tmax_reference_K"]
    else:
        Cs = _fallback_scale_sensitivity()
        assert Cs[17.0] > Cs[27.0], (
            f"fallback direction wrong: C(s=17)={Cs[17.0]:.3f} "
            f"C(s=27)={Cs[27.0]:.3f}")


# ---------------------------------------------------------------------------
# 8. Spatially correlated noise (xi = 0, 5, 10 nm, variance-fixed)
# ---------------------------------------------------------------------------
def _fallback_noise_correlation():
    """Direction check on a small grid: C(xi=10) >= C(xi=0) at fixed T."""
    X, Y = _small_grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=8.0)
    dt, Nt = 0.1, 400
    V = P.young_landscape(xx, yy, phase_k=1.0) * P.MEV_TO_NAT
    tau_nat = (TAU0_PS * (T0_K / 20.0) ** P_EXP) * P.PS_TO_NAT
    idet = int(np.argmin(np.abs(X - 60.0)))
    Cs = {}
    for xi in [0.0, 10.0]:
        rng = np.random.default_rng(31415)
        acc = np.zeros(Y.size)
        for _ in range(10):
            psi, _ = stochastic.solve2d_stochastic(
                V, psi0, dt, Nt, X, Y, tau_nat, rng,
                scale_noise=22.0, noise_xi=xi)
            p = np.abs(psi) ** 2
            acc += p[idet, :] / p[idet, :].sum()
        acc /= 10.0
        Cs[xi] = (acc.max() - acc.min()) / (acc.max() + acc.min())
    return Cs


def test_noise_correlation():
    """At fixed local noise amplitude, spatial correlation suppresses the
    differential (relative) phase that destroys the fringes: C(xi) must be
    >= C(xi=0) at every temperature and T_max must rise with xi.  Prefers
    results/noise_correlation.json; else a tiny fallback direction check."""
    d = _load_ensemble("noise_correlation.json")
    if d is not None:
        C0 = np.array(d["C_by_xi_numerical"]["0.0"])
        C5 = np.array(d["C_by_xi_numerical"]["5.0"])
        C10 = np.array(d["C_by_xi_numerical"]["10.0"])
        # strictly from 10 K upward; at the weakly-dephasing 4 K anchor the
        # xi=5 curve (N=60) is statistically indistinguishable from white
        # (N=200): bootstrap sigma(4 K) ~ 0.006-0.008, so allow a 0.01 margin.
        assert (C10 >= C0).all(), "xi=10 must never dephase more than white"
        assert (C5[1:] >= C0[1:]).all(), \
            "xi=5 must not dephase more than white for T >= 10 K"
        assert C5[0] >= C0[0] - 0.01, (
            f"4 K anchor: C(xi=5)={C5[0]:.3f} vs C(white)={C0[0]:.3f} "
            f"(must agree within the N=60 vs N=200 statistical band)")
        tm = d["Tmax_by_xi_K"]
        assert tm["0.0"] < tm["5.0"] < tm["10.0"], \
            f"T_max must rise with xi: {tm}"
        # the committed white-noise reference must be the paper's own numbers
        assert abs(C0[0] - 0.926) < 0.005 and abs(C0[1] - 0.549) < 0.005
    else:
        Cs = _fallback_noise_correlation()
        assert Cs[10.0] >= Cs[0.0], (
            f"fallback direction wrong: C(xi=0)={Cs[0.0]:.3f} "
            f"C(xi=10)={Cs[10.0]:.3f}")


def test_differential_selectivity():
    """Cross-arm leakage of the phase lens (paper Sec. 2.3): the emerged lens
    (sigma_y ~ 15 nm) is broader than the slit pitch (24 nm), so the screened
    field at the LOWER slit is a non-negligible fraction of the on-lens value.
    ratio = |V_eff(0,-12 nm)| / |V_eff(0,+12 nm)| must be in [0.25, 0.40]
    (~31% at Vg = -0.3 V) and differential_selectivity = 1 - ratio in
    [0.60, 0.75].  Prefers results/selectivity.json; else computes it live."""
    path = os.path.join(ROOT, "results", "selectivity.json")
    if os.path.exists(path):
        d = json.load(open(path))
        ratio = d["cross_arm_coupling_ratio"]
        sel = d["differential_selectivity"]
        v_up, v_low = d["V_at_upper_slit_meV"], d["V_at_lower_slit_meV"]
    else:
        lens = electrostatics.PoissonTFLens()
        v_up = float(lens.get_lens(0.0, 0.0, -0.3))
        v_low = float(lens.get_lens(0.0, -24.0, -0.3))
        ratio = abs(v_low) / abs(v_up)
        sel = 1.0 - ratio
    assert abs(v_up - (-15.0)) <= 1.0, f"V(upper slit) = {v_up} meV"
    assert v_low < 0.0, "lens is attractive: V(lower slit) must be negative"
    assert 0.25 <= ratio <= 0.40, (
        f"cross_arm_coupling_ratio = {ratio:.3f} outside [0.25, 0.40] "
        f"(V_low = {v_low:.2f} meV, V_up = {v_up:.2f} meV)")
    assert 0.60 <= sel <= 0.75, (
        f"differential_selectivity = {sel:.3f} outside [0.60, 0.75]")
    assert abs(ratio - 0.307) < 0.02, (  # pinned to the paper's ~31%
        f"ratio {ratio:.3f} must be the paper's operating-point value ~0.31")


def test_dephasing_calibration_documented():
    """The committed coherence JSON must document WHY s_phi = 22: the raw
    spec value 2.32 produces no visible dephasing, so the empirical value is a
    consistency fix anchored to the 4 K analytical point, not a physical
    prediction.  If the field is missing the test fails (paper Sec. 5 note)."""
    d = _load_ensemble()
    assert d is not None, "results/coherence_ensemble.json missing"
    note = d.get("calibration_note", "")
    assert note, "coherence_ensemble.json lacks 'calibration_note'"
    low = note.lower()
    assert "2.32" in note, "note must mention the raw spec value 2.32"
    assert "consistency fix" in low, "note must call the calibration a fix"
    assert "physical prediction" in low, (
        "note must state the calibration is NOT a physical prediction")
    assert d["scale_noise"] == 22.0


def test_no_room_temperature_claim():
    """Text-honesty regression: the manuscript must NOT claim or even mention
    room-temperature operation anywhere, and must contain the conservative
    white-noise-bound phrasing of the operating bound."""
    tex = os.path.join(ROOT, "paper", "EQLI_PhaseGate_Benchmark_2026.tex")
    assert os.path.exists(tex), f"paper source missing: {tex}"
    src = open(tex).read()
    low = src.lower()
    assert "room temperature" not in low, (
        "paper still mentions 'room temperature' (must be removed: T_max~11 K "
        "is the honest bound)")
    assert "conservative white-noise bound" in low, (
        "paper must state the 'conservative white-noise bound' framing")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed.")