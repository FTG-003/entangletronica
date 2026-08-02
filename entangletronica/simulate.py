"""Single-shot simulation pipeline: potential -> dynamics -> optics metrics.

`Simulation` ties a grid, a wave packet, a landscape and a solver into one
object with caching, so figures can be produced without recomputing.
"""

import json
import numpy as np
from . import potential as P
from . import electron
from . import gates


class Simulation:
    """Container for one physical experiment on the grid."""

    def __init__(self, X, Y, dt, Nt, landscape=None, psi0=None,
                 label="default", x0=P.X0, y0=0.0, s=P.S, k0=P.K0):
        self.X = X
        self.Y = Y
        self.dt = dt
        self.Nt = Nt
        self.landscape = landscape if landscape is not None else P.landscape_mz
        self.label = label
        self.x0, self.y0, self.s, self.k0 = x0, y0, s, k0
        xx, yy = np.meshgrid(X, Y, indexing="ij")
        self.psi0 = psi0 if psi0 is not None else electron.gaussian_packet(
            xx, yy, x0=x0, y0=y0, s=s, k0=k0)
        self._cache = {}

    # -- dynamics -------------------------------------------------------
    def run(self, force=False, **landscape_kw):
        key = ("run", tuple(sorted(landscape_kw.items())))
        if key in self._cache and not force:
            return self._cache[key]
        res = electron.run_landscape(
            self.landscape, self.X, self.Y, self.dt, self.Nt,
            psi0=self.psi0, **landscape_kw)
        self._cache[key] = res
        return res

    # -- optics ---------------------------------------------------------
    def transfer_characteristic(self, phis, splitter_k=1.0, dx=0.0, **kw):
        """Port populations vs phase-lens amplitude."""
        return gates.phase_sweep_run(
            self.X, self.Y, self.dt, self.Nt, phis,
            splitter_k=splitter_k, dx=dx, psi0=self.psi0, **kw)

    def dynamic_characteristic(self, phis, omega=0.10, A=0.5):
        return gates.dynamic_gate_run(
            self.X, self.Y, self.dt, self.Nt, omega=omega, A=A,
            phis=phis, psi0=self.psi0)

    # -- bookkeeping ----------------------------------------------------
    def metrics(self, **kw):
        psi, hist, norm, Vmev = self.run(**kw)
        Pu, Pl = gates.port_populations(hist)
        return {
            "norm": norm,
            "P_upper": Pu,
            "P_lower": Pl,
            "P_total": gates.transmission_total(hist),
            "visibility": gates.visibility(Pu, Pl),
        }

    def save(self, path):
        """Serialise the metrics of the last run to JSON."""
        m = self.metrics()
        with open(path, "w") as f:
            json.dump({"label": self.label, "metrics": m}, f, indent=2)
        return m
