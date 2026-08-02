"""Analytic models of gate-defined electrostatic potentials ("quantum lenses").

All potentials are analytic and smooth: physically, gate-defined potentials in a
2DEG are smooth on the scale of the screening length, so we deliberately avoid
hard walls. Energies are in meV, lengths in nm. The solver therefore runs with
hbar = 1 in natural units; potentials must be normalised accordingly.

Default material parameters (InGaAs 2DEG, the platform of the single-electron
semiconductor sources of [S-1]):
    m*  = 0.042 m0          effective mass
    x0  = 25 nm             wave-packet initial position
    s   = 12 nm             wave-packet spread
    k0  = 0.70 nm^-1        central momentum (ballistic reference)
"""

import numpy as np

# ----------------------------------------------------------------------------- material
MSTAR = 0.042
M0    = 9.109e-31          # kg
MEV   = 1.602e-22          # J  (1 meV)
HBAR  = 1.055e-34          # J s

# Natural units for the numerical solver: hbar = 1.
NM    = 1.0                # length unit
TIME_UNIT = (MSTAR * M0 * (NM * 1e-9) ** 2) / HBAR   # seconds per simulation time unit
MEV_TO_NAT = (MEV * TIME_UNIT**2) / (MSTAR * M0 * (NM * 1e-9) ** 2)   # meV -> solver units
# ENERGY_SCALE converts potential[meV] to solver units; time-of-flight in ps:
PS_TO_NAT = 1e-12 / TIME_UNIT

# Default wave packet (see electron.py)
X0, S, K0 = 25.0, 10.0, 1.0

# ----------------------------------------------------------------------------- helper
def _g(x, x0, s):
    """Gaussian (unnormalised)."""
    d = (x - x0) / s
    return np.exp(-0.5 * d * d)


def _erf(x):
    from scipy.special import erf
    return erf(x)


def gauss(x, x0, s, a=1.0):
    """a * exp(-0.5 ((x-x0)/s)^2)."""
    return a * _g(x, x0, s)


# ----------------------------------------------------------------------------- lens
def lens(x, y, c=(0.0, 0.0), s=12.0, a=-3.0, k=None):
    """Bipolar quantum lens: Gaussian barrier (a>0) or well (a<0).

    A lens of amplitude a in units of meV at full strength k=1.
    """
    if k is not None:
        a = a * k
    return a * _g(x, c[0], s) * _g(y, c[1], s)


def lens_gaussian_dipole(x, y, c1=(0.0, 0.0), s=12.0, a=3.0):
    """Lens with smooth radial profile (for isotropic focusing)."""
    d2 = (x - c1[0]) ** 2 + (y - c1[1]) ** 2
    return a * np.exp(-d2 / (2 * s * s))


# ----------------------------------------------------------------------------- gate geometries
def wall(x, y, xx=0.0, w=12.0, a=3.0):
    """Gate-defined barrier along y (horizontal wall at x=xx)."""
    return a * _g(x, xx, w)


def _V_naive(x, y, params):
    """Superposition of gates + contact potential (definition, unnormalised meV)."""
    V = np.zeros_like(x)
    for c, s, a in params["lenses"]:
        V += lens(x, y, c=c, s=s, a=a, k=1.0)
    for xx, w, a in params["walls"]:
        V += wall(x, y, xx=xx, w=w, a=a)
    return V


def _gate_map(x, y, params):
    """Total gate voltage landscape (meV) -- used for plotting."""
    return _V_naive(x, y, params)


# ----------------------------------------------------------------------------- beam splitter (MZ)
def beam_splitter(x, y, x0=60.0, y0=0.0, s=10.0, a=3.0, k=1.0):
    """Symmetric Gaussian barrier on the symmetry axis: the 50/50 splitter.

    A positive (repulsive) barrier centred on the propagation axis splits the
    incoming wave front into two counter-rotating halves (quantum point-contact
    analogue in 2D). Amplitude a in meV at full strength.
    """
    return a * k * _g(x, x0, s) * _g(y, y0, s)


# ----------------------------------------------------------------------------- phase shifter
def phase_shifter(x, y, x0=100.0, y0=0.0, s=12.0, a=3.0, k=1.0):
    """Gaussian well (a<0) on the lower arm: electrostatic phase shifter."""
    return a * k * _g(x, x0, s) * _g(y, y0, s)


# ----------------------------------------------------------------------------- convenient landscape
def landscape_mz(x, y, Vg=0.0, splitter_k=1.0, phase_k=1.0, dx=0.0, Vwalls=None):
    """Mach-Zehnder landscape (geometry for k0 = 1.0, domain x:[-80,272], y:[-96,96]).

    Components (strengths in meV):
      * inlet guide:   wall at x=0    (a=+30)
      * beam splitter: lens at (70, 0) (a=+150, splitter_k)  -- on-axis repulsive
      * phase lens:    lens at (140,-25) (a=-25, phase_k)    -- lower arm
      * outlet guide:  wall at x=200  (a=+30)

    Gate voltages in meV; the solver divides by MEV_TO_NAT.
    """
    V = np.zeros_like(x)
    V += wall(x, y, xx=0.0, w=8.0, a=30.0)
    V += beam_splitter(x, y, x0=70.0, y0=0.0, s=8.0, a=150.0, k=splitter_k)
    V += phase_shifter(x, y, x0=140.0, y0=-25.0, s=10.0, a=-25.0, k=phase_k)
    V += wall(x, y, xx=200.0, w=8.0, a=30.0)
    if Vwalls is not None:
        V += Vwalls(x, y)
    return V


def landscape_double_slit(x, y, Vg=0.0, barrier_k=1.0, lens_k=1.0):
    """Double-slit + Fourier lens.

      * barrier at x=90 (a=+150, barrier_k): two apertures at y=+-30
      * lens at x=140  (a=-25, lens_k): focusing lens (Gaussian well)
    """
    V = np.zeros_like(x)
    V += wall(x, y, xx=0.0, w=8.0, a=30.0)
    V += barrier_k * 150.0 * (gauss(x, 90.0, 8.0) * (1.0 - gauss(y, -30.0, 6.0)) *
                              (1.0 - gauss(y, 30.0, 6.0)))
    V += lens(x, y, c=(140.0, 0.0), s=14.0, a=-25.0 * lens_k)
    V += wall(x, y, xx=200.0, w=8.0, a=30.0)
    return V


def landscape_focus(x, y, Vg=0.0, lens_k=1.0):
    """Single focusing lens (section 4.1)."""
    V = np.zeros_like(x)
    V += wall(x, y, xx=0.0, w=8.0, a=30.0)
    V += lens(x, y, c=(90.0, 0.0), s=14.0, a=-25.0 * lens_k)
    V += wall(x, y, xx=200.0, w=8.0, a=30.0)
    return V


# ----------------------------------------------------------------------------- dynamic phase sweeps
def phase_sweep(x, y, t, phi, omega=0.10, A=0.0, c=(100.0, -20.0), s=12.0, a=-3.0):
    """Lens modulated in amplitude: A*sin(omega*t + phi) added to the base well.

    Returns (V, phase) where phase is the total applied modulation in meV.
    """
    mod = A * np.sin(omega * t + phi)
    V = lens(x, y, c=c, s=s, a=a * (1.0 + mod / abs(a)) if abs(a) > 0 else 0.0)
    return V, mod
