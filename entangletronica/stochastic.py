"""Stochastic dephasing solver: split-step Fourier with a delta-correlated
thermal potential.

Physical model (Sec. 2.4 of the paper)
--------------------------------------
Thermal / disorder-induced dephasing is modelled as a real, zero-mean,
delta-correlated stochastic potential

    <dV(r, t) dV(r', t')> = hbar^2 / (2 m* tau_phi) * delta(r-r') * delta(t-t')

In the solver's natural units (hbar = m* = 1, lengths in nm, time in
TIME_UNIT seconds) the discretised variance per grid cell per time step is

    sigma^2 = scale_noise^2 / (2 * tau_phi * dx * dy * dt)

where ``tau_phi`` is expressed in natural time units (ps * PS_TO_NAT) and
``scale_noise`` is an empirical calibration factor (see
scripts/ensemble_coherence.py): the spatial overlap of the wave packet with a
white-noise field reduces the *effective* dephasing rate relative to the
phenomenological 1/tau_phi, so the noise amplitude is calibrated against the
low-temperature analytical visibility.

Each split-step is exactly unitary (real potentials), so the norm is
conserved to round-off even in the noisy case.
"""

import numpy as np


def noise_variance(tau_phi, dx, dy, dt, scale_noise=1.0):
    """Per-cell variance of dV in solver units (natural units, hbar=m*=1)."""
    return scale_noise ** 2 / (2.0 * tau_phi * dx * dy * dt)


def dephasing_noise(shape, tau_phi, dx, dy, dt, rng, scale_noise=1.0):
    """One realisation of the delta-correlated noise field on the grid."""
    sigma = np.sqrt(noise_variance(tau_phi, dx, dy, dt, scale_noise))
    return sigma * rng.standard_normal(shape)


def solve2d_stochastic(V, psi0, dt, Nt, X, Y, tau_phi, rng,
                       scale_noise=1.0, report=False):
    """Evolve psi0 under V(x, y) plus delta-correlated thermal noise.

    Parameters
    ----------
    V : ndarray (Nx, Ny) -- deterministic potential in solver units
    psi0 : ndarray (Nx, Ny) -- initial wave function
    dt, Nt : time step and number of steps
    X, Y : 1-D coordinate arrays (for k-space construction)
    tau_phi : dephasing time in NATURAL units (time / TIME_UNIT)
    rng : numpy Generator used for the noise draws
    scale_noise : empirical calibration of the noise amplitude
    report : print the final norm

    Returns
    -------
    psi : ndarray (Nx, Ny) -- wave function on the grid (axis0 = x, axis1 = y)
    norm : final norm (~1, split-step is unitary)
    """
    Nx, Ny = psi0.shape
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]

    # reciprocal-space coordinates (axis 0 = x, axis 1 = y)
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    opK = np.exp(-0.5j * K2 * dt)

    sigma = np.sqrt(noise_variance(tau_phi, dx, dy, dt, scale_noise))
    psi = psi0.copy()
    for _ in range(Nt):
        dV = sigma * rng.standard_normal((Nx, Ny))
        opV = np.exp(-0.5j * (V + dV) * dt)
        psi = opV * psi
        psi = np.fft.ifft2(opK * np.fft.fft2(psi))
        psi = opV * psi

    norm = np.sum(np.abs(psi) ** 2)
    if report:
        print(f"[solve2d_stochastic] final norm = {norm:.12f} (unitarity ok if ~1)")
    return psi, norm
