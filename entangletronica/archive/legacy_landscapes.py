"""Archived legacy landscapes (dead code, kept for provenance).

These two analytic landscapes were the toy-model precursors of the
Poisson--Thomas--Fermi lens pipeline.  They are NOT used by any script,
test, figure, or CLI entry point of the active code: a ``grep -r`` for
``landscape_double_slit`` / ``landscape_focus`` matches only this module
and its own docstrings.

Moved here from ``potential.py`` (2026 corrections kit, item B3) to keep
the active module free of dead code.  They are imported on demand only.
"""

import numpy as np

from ..potential import wall, gauss, lens


def landscape_double_slit(x, y, Vg=0.0, barrier_k=1.0, lens_k=1.0):
    """Double-slit + Fourier lens (UNUSED / superseded, archived).

    Dead code: the published pipeline drives the Poisson--TF ``young_landscape``
    (see potential.py), so this double-slit-with-lens landscape and the
    ``landscape_focus`` variant below are retained only as an archived
    black-board sketch. No test, figure, or CLI entry point reaches them.

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
    """Single focusing lens (section 4.1). (UNUSED / superseded, archived).

    Dead code, same provenance as ``landscape_double_slit``: unused by the
    shipped model (which uses ``young_landscape``), unreferenced anywhere in
    the package, tests, or scripts. Retained only as an archived sketch.
    """
    V = np.zeros_like(x)
    V += wall(x, y, xx=0.0, w=8.0, a=30.0)
    V += lens(x, y, c=(90.0, 0.0), s=14.0, a=-25.0 * lens_k)
    V += wall(x, y, xx=200.0, w=8.0, a=30.0)
    return V
