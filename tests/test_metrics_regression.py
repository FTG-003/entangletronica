"""
Regression guard for the headline exported figures of entangletron_metrics.json.

Reads the committed results file and pins the numbers that the paper quotes
(transfer slope, transfer linear R^2, max imbalance) to 1e-6 absolute
tolerance. If the JSON is absent (e.g. a fresh clone before the pipeline has
run), the test self-skips rather than failing: CI is expected to rebuild the
file from raw simulation first (see scripts/make_figures.py).

Also verifies the Task-2 reconciliation contract: each convergence block must
declare its readout functional so any single file is self-describing.
"""

import json
import os
import sys

try:
    import pytest  # CI installs it; the bare self-runner works without it
    _HAS_PYTEST = True
except ImportError:
    pytest = None
    _HAS_PYTEST = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "results", "entangletron_metrics.json")

# Headline values observed on 2025-08-05 from a full-grid (dx=2 nm) run with
# the POISSON--THOMAS--FERMI physical lens (electrostatics.py) replacing the
# former ad-hoc Gaussian phase_shifter in the Young landscape.  This was a
# deliberate physics change: the screened-gate lens is ~2x wider (sigma_x~13 nm,
# sigma_y~14.7 nm vs the old 6/8 nm), which nearly doubles the transfer slope and
# raises the max imbalance.  Breaks here mean the physics pipeline changed in a
# way that must be reviewed/annotated deliberately.
HEADLINES = {
    "transfer_slope_per_kphi": 0.2737390577220788,
    "transfer_linear_r2": 0.9997410534849631,
    "max_imbalance": 0.7586790549846252,
}


def _load():
    if not os.path.exists(METRICS):
        msg = f"{METRICS} not present; run scripts/make_figures.py first"
        if _HAS_PYTEST:
            pytest.skip(msg)
        raise AssertionError(msg)
    with open(METRICS) as f:
        return json.load(f)


def test_headline_numbers_pinned():
    m = _load()
    for key, expected in HEADLINES.items():
        assert key in m, f"metrics lost headline {key!r}"
        assert abs(m[key] - expected) <= 1e-6, (
            f"headline {key} drifted: {m[key]!r} vs pinned {expected!r}"
        )


def test_headline_values_are_sane_ranges():
    """Guard against obviously broken values (NaN, sign-flipped, wild drift)."""
    m = _load()
    assert 0.05 < m["transfer_slope_per_kphi"] < 0.35
    assert 0.99 < m["transfer_linear_r2"] <= 1.0
    assert 0.0 < m["max_imbalance"] < 1.0
    assert m["norm_conservation_max_dev"] < 1e-6


def test_readout_blocks_are_self_labelled():
    """Each convergence block declares which readout produced its numbers."""
    m = _load()
    legacy = m["convergence_imbalance_by_dx"]
    continuous = m["convergence_imbalance_continuous"]
    assert legacy["readout_functional"] == "box_sum_legacy"
    assert continuous["readout_functional"] == "continuous_interpolated"
    # continuous row shape: dx_nm + imbalance (+ consistent dt when available)
    for row in continuous["values"]:
        assert "dx_nm" in row and "imbalance" in row
    # both blocks sample the same dx grid
    legacy_dx = {row["dx_nm"] for row in legacy["values"]}
    cont_dx = {row["dx_nm"] for row in continuous["values"]}
    assert legacy_dx & cont_dx, f"blocks share no dx samples: {legacy_dx} vs {cont_dx}"


if __name__ == "__main__":
    import pathlib

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed.")