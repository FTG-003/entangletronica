"""Analytic models of gate-defined electrostatic potentials ("quantum lenses").

All potentials are analytic and smooth: physically, gate-defined potentials in a
2DEG are smooth on the scale of the screening length, so we deliberately avoid
hard walls. Energies are in meV, lengths in nm. The solver therefore runs with
hbar = 1 in natural units; potentials must be normalised accordingly.

Default material parameters (InGaAs 2DEG):
    m* = 0.042 m0    effective mass (used for the unit conversions below)
    X0 = 25 nm       wave-packet initial position
    S  = 10 nm       wave-packet spread
    K0 = 1.0 nm^-1   central momentum (package default; the published
                     experiment scripts override it to k0 = 0.2 nm^-1,
                     see scripts/entangletron_experiment.py)
"""

import numpy as np
from .electrostatics import PoissonTFLens

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
    """Gaussian well (a<0) on the lower arm: electrostatic phase shifter.

    Ad-hoc analytic gate, superseded by the Poisson--Thomas--Fermi lens
    (:class:`electrostatics.PoissonTFLens`) in the Young landscape below;
    retained for the Mach--Zehnder landscape and the archived sketches.
    """
    return a * k * _g(x, x0, s) * _g(y, y0, s)


# ----------------------------------------------------------------------------- Young landscape (main pipeline)
# Physical lens of Sec. 2.2: gate finger (w=20, t=10, d=20 nm, eps_r=13.9)
# screened by the 2DEG; amplitude calibrated to V0 = -15 meV at Vg = -0.3 V
# (linear coupling 50 meV/V in the shallow-lens regime |Vg| <= 0.3 V).
VG_PER_KPHI = -0.3    # [V] per unit phase_k (negative: attractive lens)
_PHYS_LENS = PoissonTFLens()


def young_landscape(x, y, Vg=0.0, phase_k=0.0, barrier_k=1.0, barrier_a=12.0,
                    slit_y=12.0, barrier_x=60.0, lens_x=68.0, lens_y=12.0):
    """Double-slit + Poisson--Thomas--Fermi phase lens (main pipeline).

    Components (strengths in meV):
      * entrance/exit caps: wall at x=0 and x=160 (a=+15)
      * double-slit barrier: Gaussian sheet at x=barrier_x (a=+12*barrier_k)
        with transparent apertures of width 4 nm at y=+-slit_y
      * phase lens: screened gate finger (PoissonTFLens) at (lens_x, lens_y)
        behind the upper slit, driven by Vg_eff = Vg + VG_PER_KPHI * phase_k

    The lens depth factor ``phase_k`` is the legacy knob of the toy model;
    the physical control parameter is the gate voltage Vg (1 unit of phase_k
    <-> -0.3 V <-> -15 meV, Sec. 2.2 of the paper).  Returns meV.
    """
    V = np.zeros_like(x)
    V += wall(x, y, xx=0.0, w=6.0, a=15.0)
    V += barrier_k * barrier_a * gauss(x, barrier_x, 6.0) * \
         (1.0 - gauss(y, -slit_y, 4.0)) * (1.0 - gauss(y, slit_y, 4.0))
    V += _PHYS_LENS.get_lens(x - lens_x, y - lens_y, Vg + VG_PER_KPHI * phase_k)
    V += wall(x, y, xx=160.0, w=6.0, a=15.0)
    return V


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


# ----------------------------------------------------------------------------- dynamic phase sweeps
def phase_sweep(x, y, t, phi, omega=0.10, A=0.0, c=(100.0, -20.0), s=12.0, a=-3.0):
    """Lens modulated in amplitude: A*sin(omega*t + phi) added to the base well.

    Returns (V, phase) where phase is the total applied modulation in meV.
    """
    mod = A * np.sin(omega * t + phi)
    V = lens(x, y, c=c, s=s, a=a * (1.0 + mod / abs(a)) if abs(a) > 0 else 0.0)
    return V, mod
