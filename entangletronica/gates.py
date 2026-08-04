"""Quasi-static gate sweeps: charge conservation, interference visibility.

The Mach-Zehnder of section 4.2 is swept by tilting the phase lens amplitude.
For every gate configuration we evolve the same electron and measure the
interference at the two output ports; the result is an oscillating transfer
characteristic whose visibility is the engineering metric of the gate.
"""

import numpy as np
from . import electron, potential as P

# ----------------------------------------------------------------------------- output ports
def port_populations(hist):
    """Upper (U) and lower (L) port populations from a simulation histogram."""
    return hist["upper"], hist["lower"]


def visibility(Pu, Pl):
    """Interference visibility V = (Pu-Pl)/(Pu+Pl) per run; and contrast on a sweep."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Pu + Pl > 0, (Pu - Pl) / (Pu + Pl), 0.0)


def sweep_visibility(vis):
    """Contrast of an oscillation: (max-min)/(max+min) of the port imbalance."""
    v = np.asarray(vis)
    return (v.max() - v.min()) / (v.max() + v.min()) if v.max() + v.min() > 0 else 0.0

# ----------------------------------------------------------------------------- charge conservation
def charge_conservation(psi):
    return np.sum(np.abs(psi) ** 2)


def transmission_total(hist):
    """Total probability reaching the detector zone (leakage metric).

    ``centre`` is a subset of ``upper ∪ lower`` (the histogram splits at
    y=0), so summing all three would double-count the central band. Use only
    the two disjoint bins for the total.
    """
    return hist["upper"] + hist["lower"]

# ----------------------------------------------------------------------------- phase sweep
def phase_sweep_run(X, Y, dt, Nt, phis, splitter_k=1.0, phase_k=1.0, dx=0.0,
                    psi0=None, x0=P.X0, s=P.S, k0=P.K0, Vwalls=None,
                    y0=0.0):
    """Run the MZ for a list of phase-lens depth factors ``phis``.

    ``phis`` is the dimensionless factor ``phase_k`` multiplying the -25 meV
    phase lens (phase_k = 1 <-> full -25 meV depth).

    Returns dict with populations, visibility and the trajectory of the final
    wave function (for movies / snapshots).
    """
    if psi0 is None:
        xx, yy = np.meshgrid(X, Y, indexing="ij")
        psi0 = electron.gaussian_packet(xx, yy, x0=x0, y0=y0, s=s, k0=k0)

    results = []
    for ph in phis:
        psi, hist, norm, _ = electron.run_landscape(
            P.landscape_mz, X, Y, dt, Nt, psi0=psi0,
            Vg=0.0, splitter_k=splitter_k, phase_k=ph,
            dx=dx, Vwalls=Vwalls,
        )
        Pu, Pl = port_populations(hist)
        results.append({
            "phase": ph,
            "upper": Pu, "lower": Pl, "norm": norm,
            "psi": psi, "hist": hist,
        })
    return results

# ----------------------------------------------------------------------------- dynamic (in-flight) phase gate
def dynamic_gate_run(X, Y, dt, Nt, omega=0.10, A=0.5, phis=np.linspace(0, 2*np.pi, 9, endpoint=False),
                     psi0=None, x0=P.X0, s=P.S, k0=P.K0, y0=0.0):
    """Time-modulated phase lens: A*sin(omega*t + phi) added to the static well.

    This is the "dynamic electrostatic gate": the phase acquired by the lower arm
    is programmed in flight. Returns per-phi final populations.
    """
    if psi0 is None:
        xx, yy = np.meshgrid(X, Y, indexing="ij")
        psi0 = electron.gaussian_packet(xx, yy, x0=x0, y0=y0, s=s, k0=k0)

    Vmev0 = P.landscape_mz(*np.meshgrid(X, Y, indexing="ij"), Vg=0.0, splitter_k=1.0, phase_k=1.0)
    V0 = Vmev0 * P.MEV_TO_NAT

    # base potential without the phase lens (we add the modulated lens by hand)
    Vbase_mev = P.landscape_mz(*np.meshgrid(X, Y, indexing="ij"), Vg=0.0, splitter_k=1.0, phase_k=0.0)
    Vbase = Vbase_mev * P.MEV_TO_NAT

    results = []
    for phi in phis:
        # build time-dependent potential on the fly
        def Vt(t):
            Vmod, _ = P.phase_sweep(*np.meshgrid(X, Y, indexing="ij"), t, phi,
                                    omega=omega, A=A, c=(140.0, -25.0), s=10.0, a=-25.0)
            return Vbase + Vmod * P.MEV_TO_NAT

        psi = psi0.copy()
        for i in range(Nt):
            t = i * dt
            V = Vt(t)
            psi = electron.step(psi, V, dt, X, Y)  # single step (defined below)
        Pu, Pl = port_populations(electron._histogram(psi, X, Y))
        results.append({"phase": float(phi), "upper": Pu, "lower": Pl,
                        "norm": float(np.sum(np.abs(psi) ** 2)), "psi": psi})
    return results
