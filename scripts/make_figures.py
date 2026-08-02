"""Figure pipeline (single entry point).

Delegates to entangletron_experiment.py so there is exactly one source of
truth for the device parameters and figures.

    python scripts/make_figures.py     # same as entangletron_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from entangletron_experiment import main
    main()
