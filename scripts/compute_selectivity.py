#!/usr/bin/env python3
"""Cross-arm coupling / differential selectivity of the phase lens (Sec. 2.3).

The screened lens (PoissonTFLens) is centred on the upper slit
(lens_x=68 nm, lens_y=+12 nm in young_landscape).  Because the emerged
lens is broad compared with the slit pitch (sigma_y ~ 15 nm vs 24 nm),
a fraction of the gate field leaks onto the *lower* slit, coupling the
gate to both arms.  This script quantifies that leakage:

    cross_arm_coupling_ratio  = |V_eff(lower slit)| / |V_eff(upper slit)|
    differential_selectivity  = 1 - cross_arm_coupling_ratio

at the operating point Vg = -0.3 V (V0 = -15 meV), using the raw
Poisson--Thomas--Fermi lens (no barrier, no readout).

Run:  python3 scripts/compute_selectivity.py
Out:  results/selectivity.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from entangletronica.electrostatics import PoissonTFLens

# Operating point (Sec. 2.2): Vg = -0.3 V <-> V_eff(0,0) = -15 meV
VG = -0.3
# Device geometry (young_landscape defaults): lens behind the UPPER slit.
LENS_X, LENS_Y = 68.0, 12.0      # nm (lens centre = upper slit centre)
SLIT_Y = 12.0                    # nm (slit pitch 24 nm -> slits at +-12)


def main():
    lens = PoissonTFLens()
    # Lens-frame coordinates: upper slit centre = (0,0), lower = (0,-24).
    v_up = float(lens.get_lens(0.0, 0.0, VG))
    v_low = float(lens.get_lens(0.0, -2 * SLIT_Y, VG))
    ratio = abs(v_low) / abs(v_up)
    out = {
        "Vg_V": VG,
        "V_at_upper_slit_meV": v_up,
        "V_at_lower_slit_meV": v_low,
        "cross_arm_coupling_ratio": ratio,
        "differential_selectivity": 1.0 - ratio,
        "slit_pitch_nm": 2 * SLIT_Y,
        "lens_centre_upper_slit": {"x_nm": LENS_X, "y_nm": LENS_Y},
        "note": (
            "V_eff sampled at the slit centres in the lens frame "
            "(upper (0,0), lower (0,-24 nm)). ratio = |V_lower|/|V_upper|; "
            "differential_selectivity = 1 - ratio. The lens is not "
            "single-slit selective: it also phase-shifts the lower arm."
        ),
    }
    path = os.path.join(ROOT, "results", "selectivity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
