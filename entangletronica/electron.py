"""(x, y, t) splitting of the time-dependent Schroedinger equation.

We evolve the 2D wave function on a Cartesian grid with a split-step operator:

    psi(t+dt) = exp(-i V dt/2) FFT^-1 [ exp(-i k^2 dt/2) FFT [ exp(-i V dt/2) psi ] ]

Natural units: hbar = m = 1 (see potential.py for conversions).
The scheme is exactly unitary (each step is the product of two unitary operators),
so the norm is conserved up to round-off -- we verify it after every run.

Default wave packet: minimum-uncertainty Gaussian in (x, y), initial momentum
along +x, localised in the y direction (a "flying electron").
"""

import numpy as np
from . import potential as P

# ----------------------------------------------------------------------------- wave packet
def gaussian_packet(x, y, x0=P.X0, y0=0.0, s=P.S, k0=P.K0):
    """Minimum-uncertainty Gaussian packet, momentum along +x (normalised)."""
    kx = k0
    ky = 0.0
    psi = (np.exp(-0.25 * ((x - x0) / s) ** 2 - 0.25 * ((y - y0) / s) ** 2) *
           np.exp(1j * (kx * (x - x0) + ky * (y - y0))))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi

# ----------------------------------------------------------------------------- solver
def solve2d(V, psi0, dt, Nt, X, Y, report=True):
    """Evolve psi0 for Nt steps of dt under potential V(x, y).

    Parameters
    ----------
    V : ndarray (Ny, Nx) -- potential in solver units, constant in time
    psi0 : ndarray (Ny, Nx) -- initial wave function
    dt, Nt : time step and number of steps
    X, Y : 1-D coordinate arrays (for k-space construction)

    Returns
    -------
    psi : ndarray (Nx, Ny) -- wave function on the grid (axis0 = x, axis1 = y)
    hist : dict with |psi|^2 integrated over regions (upper/lower arm, etc.)
    norm : final norm (should be ~1)
    """
    Nx, Ny = psi0.shape
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]

    # reciprocal-space coordinates (axis 0 = x, axis 1 = y)
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2

    # operators
    opV = np.exp(-0.5j * V * dt)
    opK = np.exp(-0.5j * K2 * dt)

    psi = psi0.copy()
    norm = np.sum(np.abs(psi) ** 2)

    for _ in range(Nt):
        psi = opV * psi
        psi = np.fft.ifft2(opK * np.fft.fft2(psi))
        psi = opV * psi

    norm = np.sum(np.abs(psi) ** 2)
    if report:
        print(f"[solve2d] final norm = {norm:.12f} (unitarity ok if ~1)")

    # detection regions (physical positions in nm)
    hist = _histogram(psi, X, Y)
    return psi, hist, norm

def _histogram(psi, X, Y):
    """Integrated probability in the detector regions (upper/lower/centre)."""
    p = np.abs(psi) ** 2
    Nx, Ny = p.shape
    x_sel = np.where((X >= X[-1] - 25) & (X <= X[-1]))[0]
    if len(x_sel) == 0:
        return {"upper": 0.0, "lower": 0.0, "centre": 0.0}
    psel = p[x_sel, :]
    # split at y=0
    mid = int(np.searchsorted(Y, 0.0))
    upper = float(np.sum(psel[:, mid:])) if mid < Ny else 0.0
    lower = float(np.sum(psel[:, :mid])) if mid > 0 else 0.0
    centre = float(np.sum(psel[:, mid - 2:mid + 2])) if 1 < mid < Ny - 1 else 0.0
    return {"upper": upper, "lower": lower, "centre": centre}

def step(psi, V, dt, X, Y):
    """One split-step evolution under time-independent V (for dynamic sweeps)."""
    Nx, Ny = psi.shape
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    opV = np.exp(-0.5j * V * dt)
    opK = np.exp(-0.5j * K2 * dt)
    psi = opV * psi
    psi = np.fft.ifft2(opK * np.fft.fft2(psi))
    psi = opV * psi
    return psi


# ----------------------------------------------------------------------------- convenience: full run
def run_landscape(landscape, X, Y, dt, Nt, psi0=None, t0=0.0, Vg=0.0,
                  **landscape_kw):
    """Build the potential from a landscape callable, evolve, return everything."""
    if psi0 is None:
        psi0 = gaussian_packet(*np.meshgrid(X, Y, indexing="ij"))
    Vmev = landscape(*np.meshgrid(X, Y, indexing="ij"), Vg=Vg, **landscape_kw)
    V = Vmev * P.MEV_TO_NAT
    psi, hist, norm = solve2d(V, psi0, dt, Nt, X, Y)
    return psi, hist, norm, Vmev
