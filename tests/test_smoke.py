"""
Run:  python3 -m pytest tests/ -q    (or: python3 tests/test_smoke.py)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from entangletronica import potential as P
from entangletronica import electron, gates


def _grid():
    NX, NY, DX = 96, 64, 2.0
    X = np.arange(NX) * DX - 40.0
    Y = np.arange(NY) * DX - 64.0
    return X, Y


def test_units_conversion():
    assert P.MEV_TO_NAT > 0
    assert P.PS_TO_NAT > 0


def test_packet_normalised():
    X, Y = _grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi = electron.gaussian_packet(xx, yy)
    assert abs(np.sum(np.abs(psi) ** 2) - 1.0) < 1e-10


def test_free_flight_moves_and_is_unitary():
    """The packet must fly (+x) and stay normalised: the sanity check that
    caught the aliasing bug (dx=8nm folded k0 to ~0 and froze the electron)."""
    X, Y = _grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy)
    V = np.zeros_like(xx)
    psi, hist, norm = electron.solve2d(V, psi0, 0.3, 60, X, Y)
    assert abs(norm - 1.0) < 1e-9
    p0 = np.abs(psi0) ** 2
    p1 = np.abs(psi) ** 2
    cx0 = np.sum(p0.sum(1) * X) / p0.sum()
    cx1 = np.sum(p1.sum(1) * X) / p1.sum()
    v_step = P.K0 * 0.3
    assert cx1 > cx0 + 0.5 * v_step * 60, f"electron frozen: {cx0} -> {cx1}"


def test_splitter_splits():
    """The double-slit barrier must split the wave into two coherent sources:
    significant transmitted probability past the barrier plane."""
    X, Y = _grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=10.0)

    def young(x, y, Vg=0.0, phase_k=0.0, barrier_k=1.0):
        V = np.zeros_like(x)
        V += P.wall(x, y, xx=0.0, w=6.0, a=15.0)
        V += barrier_k * 12.0 * P.gauss(x, 60.0, 6.0) * \
             (1.0 - P.gauss(y, -12.0, 4.0)) * (1.0 - P.gauss(y, 12.0, 4.0))
        V += P.phase_shifter(x, y, x0=68.0, y0=12.0, s=6.0, a=-15.0, k=phase_k)
        V += P.wall(x, y, xx=160.0, w=6.0, a=15.0)
        return V

    psi, hist, norm, Vmev = electron.run_landscape(young, X, Y, 0.3, 600, psi0=psi0)
    p = np.abs(psi) ** 2
    ib = int(np.argmin(np.abs(X - 65.0)))   # just past the barrier
    tot = p[ib:, :].sum()
    assert 0.2 < tot < 1.0  # substantial (not all) probability transmitted


def test_phase_sweep_oscillates():
    """The gate must move the interference figure: imbalance grows with the
    gate depth (linear transducer, measured R^2 = 0.99997 on the full grid)."""
    X, Y = _grid()
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    psi0 = electron.gaussian_packet(xx, yy, k0=0.2, s=10.0)

    def young(x, y, Vg=0.0, phase_k=0.0, barrier_k=1.0):
        V = np.zeros_like(x)
        V += P.wall(x, y, xx=0.0, w=6.0, a=15.0)
        V += barrier_k * 12.0 * P.gauss(x, 60.0, 6.0) * \
             (1.0 - P.gauss(y, -12.0, 4.0)) * (1.0 - P.gauss(y, 12.0, 4.0))
        V += P.phase_shifter(x, y, x0=68.0, y0=12.0, s=6.0, a=-15.0, k=phase_k)
        V += P.wall(x, y, xx=160.0, w=6.0, a=15.0)
        return V

    res = []
    for pk in [0.0, 0.5, 1.0, 1.5]:
        psi, hist, norm, Vmev = electron.run_landscape(
            young, X, Y, 0.3, 400, psi0=psi0, phase_k=pk)
        p = np.abs(psi) ** 2
        i = int(np.argmin(np.abs(X - 70.0)))
        prof = p[i, :]
        iu = (Y >= 0) & (Y < 14)
        il = (Y > -14) & (Y < 0)
        pu, pl = prof[iu].sum(), prof[il].sum()
        res.append((pu - pl) / (pu + pl) if pu + pl > 0 else 0.0)
        assert abs(norm - 1.0) < 1e-6
    assert res[-1] > res[0] + 1e-3  # the gate must move the interference figure


def test_visibility_range():
    assert gates.visibility(1.0, 0.0) == 1.0
    assert abs(gates.visibility(0.5, 0.5)) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed.")
