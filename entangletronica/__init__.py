"""Entangletronica -- single-electron flying-qubit quantum simulations.

Modules
-------
potential : analytic electrostatic potentials of gate-defined quantum lenses
electron  : (x, y, t) splitting of the 2D Schroedinger equation, unitarity check
gates     : quasi-static gate sweeps with charge conservation and visibility metrics
simulate  : single-shot simulation pipeline (potential -> dynamics -> optics)
"""
from . import potential, electron, gates, simulate

__version__ = "0.1.0"

__all__ = ["potential", "electron", "gates", "simulate", "__version__"]
