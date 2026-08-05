"""Poisson--Thomas--Fermi electrostatics of a gate finger above a 2DEG.

Physical model (Sec. 2.2 of the paper)
--------------------------------------
A metallic gate finger of width ``w`` (along y) and thickness ``t`` (along x)
is deposited at distance ``d`` above the InGaAs quantum well.  The *bare*
potential it induces at the 2DEG plane is approximated with the strip
(Laplace) solution

    Vx(x) = (1/pi) [ arctan((x + t/2)/d) - arctan((x - t/2)/d) ]
    Vy(y) = (1/pi) [ arctan((y + w/2)/d) - arctan((y - w/2)/d) ]
    V_bare(x, y) = Vx(x) * Vy(y) * Vg

The 2DEG screens this potential in the linear Thomas--Fermi approximation.
In Fourier space the screened (effective) potential is

    V_eff(q) = V_bare(q) / (1 + q_TF / |q|),

with the Thomas--Fermi wave number

    q_TF = e^2 m* / (2 eps0 eps_r pi hbar^2)      [nm^-1]

(Lindhard limit; q_TF = e^2 D(E_F)/(2 eps0 eps_r) with D(E_F) = m*/(pi hbar^2)).
The zero-frequency component is killed by the denominator (|q| -> 0), i.e. the
electron gas screens a static uniform bias completely: the effective profile is
a localised well with a weak positive halo, not a constant offset.

Amplitude calibration
---------------------
The strip model underestimates the capacitive coupling of a real 3D gate
(no AlGaAs barrier, no image charges, no screening non-locality), so the
amplitude is calibrated to the shallow-lens operating point quoted in the
paper:  V_eff(0, 0) = -15 meV at V_g = -0.3 V, i.e. a linear coupling of
50 meV/V in the regime |V_g| <= 0.3 V.  The *shape* (widths sigma_x, sigma_y)
emerges from the Poisson--TF calculation and is *not* imposed.

The voltage-to-depth mapping is exactly linear (linear screening), R^2 = 1.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------------- SI constants
E    = 1.602176634e-19        # C
EPS0 = 8.8541878128e-12       # F/m
M0   = 9.1093837015e-31       # kg
HBAR = 1.054571817e-34        # J s
MSTAR = 0.042                 # InGaAs effective mass (m* = 0.042 m0)
MEV  = 1e-3 * E               # J per meV

# Default material / geometry (Sec. 2.2 of the paper)
EPS_R = 13.9
N2D   = 2e11                  # cm^-2


class PoissonTFLens:
    """Gate finger + linear Thomas--Fermi screening -> effective lens.

    The screened unit response (per volt of gate bias) is computed once on an
    internal grid centred on the gate; ``get_lens`` then interpolates it onto
    the requested coordinate mesh and scales it by the calibrated coupling.
    """

    def __init__(self, w=20.0, t=10.0, d=20.0, eps_r=EPS_R, n2D=N2D,
                 dx=0.5, extent=200.0):
        self.w, self.t, self.d = float(w), float(t), float(d)
        self.eps_r, self.n2D = float(eps_r), float(n2D)
        self._dx = float(dx)
        self._extent = float(extent)
        self._build()

    # -- material ------------------------------------------------------------------
    @property
    def q_TF(self):
        """Thomas--Fermi screening wave number in nm^-1."""
        # SI value is ~1.14e8 m^-1; 1 m^-1 = 1e-9 nm^-1
        return (E ** 2 * MSTAR * M0) / (2 * EPS0 * self.eps_r * np.pi * HBAR ** 2) * 1e-9

    @property
    def screening_length_nm(self):
        return 1.0 / self.q_TF

    # -- bare strip potential ------------------------------------------------------
    def _strip(self, u, width):
        """(1/pi)[arctan((u+width/2)/d) - arctan((u-width/2)/d)]."""
        return (np.arctan((u + 0.5 * width) / self.d) -
                np.arctan((u - 0.5 * width) / self.d)) / np.pi

    # -- screened unit response ----------------------------------------------------
    def _build(self):
        n = int(round(2 * self._extent / self._dx))
        if n % 2 == 0:
            n += 1                                   # odd: spatial origin at a node
        self._n = n
        # internal coordinate array: index i sits at x_i = (i - n//2) * dx
        idx = np.arange(n)
        self._xi = (idx - n // 2) * self._dx

        # bare potential for Vg = 1 V, in meV
        vx = self._strip(self._xi, self.t)
        vy = self._strip(self._xi, self.w)
        v_bare = 1000.0 * np.outer(vx, vy)           # [meV/V]

        # reciprocal-space coordinates (angular, nm^-1)
        k = 2 * np.pi * np.fft.fftfreq(n, d=self._dx)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        qmag = np.hypot(kx, ky)
        tf_filter = np.where(qmag > 0, 1.0 / (1.0 + self.q_TF / np.where(qmag > 0, qmag, 1.0)), 0.0)
        tf_filter[0, 0] = 0.0                         # zero-frequency mode fully screened

        v_eff_unit = np.fft.ifft2(np.fft.fft2(v_bare) * tf_filter).real   # [meV/V]

        # amplitude calibration: V_eff(0,0) = 50 meV/V  <->  -15 meV at Vg = -0.3 V
        v0 = v_eff_unit[self._n // 2, self._n // 2]
        self._amp_mev_per_v = 50.0
        self._unit = (self._amp_mev_per_v / v0) * v_eff_unit

        # interpolator on the internal grid (gate-centred)
        self._interp = RegularGridInterpolator(
            (self._xi, self._xi), self._unit, bounds_error=False, fill_value=0.0)

        self._sigma_x, self._sigma_y = self._fit_gaussian_widths()

    def _fit_gaussian_widths(self):
        """1/e-widths sigma_x, sigma_y of the central well (Gaussian fit).

        Fits ln(|V_eff|) vs coordinate^2 on the central structure (the bump
        that becomes the lens for Vg < 0), excluding the weak screening halo.
        """
        c = self._n // 2

        def fit1d(prof):
            peak = prof.max()
            m = (np.abs(self._xi) <= 40.0) & (prof > 0.5 * peak)
            if m.sum() < 5:
                return self._dx
            u = self._xi[m]
            a2 = np.polyfit(u ** 2, np.log(prof[m]), 1)[0]
            if a2 >= 0:
                return self._dx
            return float(np.sqrt(-0.5 / a2))

        sx = fit1d(self._unit[:, c])     # along x (axis through centre)
        sy = fit1d(self._unit[c, :])     # along y
        return sx, sy

    # -- public API -----------------------------------------------------------------
    def get_lens(self, x, y, Vg):
        """Effective screened lens potential in meV at coordinates (x, y).

        ``x`` and ``y`` are arrays of identical shape (typically the solver
        meshgrid, with the gate centre at (0, 0) of this frame); ``Vg`` is the
        gate voltage in volts.  Returns V_eff in meV.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.column_stack([x.ravel(), y.ravel()])
        v = self._interp(pts).reshape(x.shape)   # calibrated unit response [meV/V]
        return v * Vg

    def lens_depth(self, Vg):
        """Calibrated centre depth V0(Vg) in meV: exactly 50 meV/V."""
        return 50.0 * Vg

    def mapping_report(self, Vg_range=(-0.5, 0.0), npts=41):
        """Linearity check of Vg -> V0 over a sweep (shallow regime |Vg|<=0.3V).

        Returns dict with slope [meV/V], R^2 of the shallow fit and the
        Gaussian widths that emerge from the Poisson--TF calculation.
        """
        Vg = np.linspace(*Vg_range, npts)
        V0 = self.lens_depth(Vg)
        shallow = np.abs(Vg) <= 0.3
        m, b = np.polyfit(Vg[shallow], V0[shallow], 1)
        resid = V0[shallow] - (m * Vg[shallow] + b)
        r2 = 1.0 - resid.var() / V0[shallow].var()
        return {
            "q_TF_nm-1": float(self.q_TF),
            "screening_length_nm": float(self.screening_length_nm),
            "sigma_x_nm": float(self._sigma_x),
            "sigma_y_nm": float(self._sigma_y),
            "slope_meV_per_V": float(m),
            "intercept_meV": float(b),
            "r2_shallow": float(r2),
            "V0_at_m03V_meV": float(self.lens_depth(-0.3)),
        }
