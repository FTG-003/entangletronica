<div align="center">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21802416.svg)](https://doi.org/10.5281/zenodo.21802416)
# Flying-Electron Interference Logic via Electrostatic Quantum Lenses

### Entangletronica

**Fabrizio Terzi** · [Entangletronica Lab](https://github.com/FTG-003)

*Reproducible split-step Schrödinger simulations of single-electron
interference in a ballistic 2DEG.*

**Status** — reproducible numerical study. **Scope** — a computing exercise whose
physics is anchored to established literature, not a claim of new experimental
observation.

</div>

---

## What this is

A cleanly-engineered, fully reproducible simulation pipeline for a **single
flying electron** passing through a **gate-defined double slit** in a ballistic
2DEG, with a programmable electrostatic gate behind one aperture that shifts the
downstream interference figure. This is a *pedagogical and engineering* study in
the spirit of the electronic-interferometry literature (see
[below](#related-work)); it does **not** claim new physics or new experimental
results.

The name **"Entangletronica" is used descriptively, not to imply many-particle
entanglement**: the physics here is *single-particle* interference, and the
"quantum lenses" are simply electrostatic potential gates. The framing is honest
about this throughout.

---

## Headline numbers — and their honest meaning

The two numbers most often quoted about this project carry very different
epistemic weight, and it matters that they are not conflated.

| Number value | What it actually is | Epistemic weight |
|---|---|---|
| **R² = 0.9997** (fit to transfer curve) | Self-consistency of a deterministic solver in a linear regime | **Low** — a well-engineered PDE solver *must* give a nearly-linear curve here. Quasi-tautological; sanity check, not discovery. |
| **Norm conserved to 1e-13** | Round-off-level unitarity of the split-step scheme | **Solver correctness** (numerics), not a device result. |

These belong to the **numerical engine**, and are reported because they
establish that the *code* is trustworthy — the necessary precondition for the
physics being worth looking at at all. They are **not the scientific
result**.

The scientific content lives in what the gate does as a transducer:
throughput a well-defined, reproducible, roughly-linear response of detector
imbalance to gate depth (see results below), which mirrors textbook double-slit
physics — expected, quantifiable, and consistent with the model.

---

## Table of contents

1. [Architecture](#architecture)
2. [Solver evidence (reproducible numerics)](#solver-evidence)
3. [Device-physics evidence](#device-physics-evidence)
4. [Reproduce](#reproduce)
5. [Conventions](#conventions)
6. [Related work](#related-work)
7. [Limitations & caveats](#limitations--caveats)

---

## Architecture

A **Young-type double-slit interferometer** for a single flying electron in a
ballistic 2DEG:

1. a Gaussian wave packet (`k₀ = 0.20 nm⁻¹`, `E = 36.3 meV`, `λ = 31.4 nm`)
   is launched into the channel;
2. a gate-defined double-slit barrier (`x = 60 nm`, apertures at
   `y = ±12 nm`) splits the wave into two coherent sources;
3. a **programmable electrostatic gate** (Poisson–Thomas–Fermi screened
   lens, `V₀ = −15 meV` at `Vg = −0.3 V`, behind the upper aperture) shifts
   the *relative phase* between the paths **in flight**;
4. the displaced interference figure is read by a two-bin detector at
   `x = 110 nm`.

Because the carrier is *flying* (ballistically transiting the device), the gate
could in principle be modulated during the transit — the conceptual motivation
for "picosecond" fast, a scale set by the ballistic transit time, *not* a
measured switching experiment.

---

## 2. Solver evidence (reproducible numerics)

- **Engine:** split-step Fourier / operator-splitting for the time-dependent
  2D Schrödinger equation (natural units `ħ = m = 1`).
- **Scheme is exactly unitary** (each substep is a product of unitary
  operators), so the norm is conserved to round-off — measured at ~1e-13.
- Grid: 140 × 80, `dx = 2 nm` — this **resolves `k₀` with 4× the Nyquist
  margin**, which is precisely the correction that makes the physics correct
  (an 8 nm grid aliases `k₀`, freezing the packet; cf. `NOTES.md`).
- **Sanity checks:** `tests/test_smoke.py` asserts unitarity, free-flight
  propagation, the double-split transmitting a substantial fraction, monotone
  gate response, and unit conversions.

These test that the *calculator is correct*, not that a device is special.

A **grid-refinement study** (`scripts/convergence_study.py`) reduces the grid
`Δx = {4, 2, 1, 0.5} nm` (with `Δt` scaled to keep the physical propagation
time constant) at a fixed gate setting. The norm is exact to round-off at
every resolution, the fringe peak settles to ≈ +10.3 nm (stable to <0.01 nm),
and the *profile* L2 error decays near second order. A summary of the
observables:

* fringe peak — converged (≈ +10.3 nm, stable <0.01 nm);
* centroid & width — converged (≈ +5.87 nm, ≈ 13.86 nm);
* profile L2 vs finest — 0.0044 → 0.0016 → 0.00027 (near 2nd order);
* two-bin imbalance read as a *continuous* functional of the profile — stable
  at ≈ +0.295 from `Δx = 2` nm already.

**Why an earlier imbalance looked unconverged.** `scripts/readout_sensitivity.py`
re-propagated exactly the same state and read the same profile two ways. A raw
lattice **box-sum** (the old observable) drifts `+0.445 → +0.376 → +0.336 →
+0.315` as the grid coarsens, because a coarse grid samples a fixed-width box
from too few lines; the **continuous functional** on the same state is stable
at `≈ +0.295` from `Δx = 2` down. The discrepancy closes exactly as the grid
refines (`+0.15 → +0.08 → +0.04 → +0.02`).

**Conclusion:** the non-convergence was an artefact of the *lattice readout*,
not of the quantum dynamics. A correct (interpolated) readout is converged at
`Δx = 2 nm`; no finer grid was needed to settle the question. We present this
as a grid-refinement *and readout-sensitivity* study, `Δx = 0.25 nm` deferred
(reserved for an actual physics question, not a readout artefact). Result
produced by `readout_sensitivity.py`, figure `fig6_readout_profile.pdf`.

---

## 3. Device-physics evidence

| Observable | Value | Interpretation |
|---|---|---|
| Detector imbalance (max) | **0.76** (at Vg = −0.75 V) | Interference redistributes probability between bins → a working readout |
| Transfer sensitivity | **S_V ≈ −0.91 V⁻¹** (shallow window) | Phase–voltage transduction, consistent with the 50 meV/V electrostatics |
| Rough linearity of the transfer curve | R² ≈ 0.9997 | Regime statement, not a property of the device — see note below |
| Full-line fringe contrast (coherent limit) | ≈ 1.0 | Profile tails vanish; dephasing fills them → C(T) is the coherence observable |
| Lens widths (Poisson–TF) | σx ≈ 13 nm, σy ≈ 15 nm | Emerge from screening (q_TF = 0.11 nm⁻¹), not assumed |
| Input parameters | k₀ = 0.20 nm⁻¹, E = 36.3 meV, λ = 31.4 nm | Plausible *within* an InGaAs 2DEG window, chosen for a clean plot — see [caveats](#limitations--caveats) |

The device operates as a **linear-ish transduction**: a gate parameter in, an
interference-related output, over a physically sensible operating range.
This is the *expected* behaviour of double-slit interference, reproduced
cleanly — not a discovery.

> **Why the transfer looks so linear (measured, not assumed).** Each gate unit
> (`phase_k`, 1 ↔ Vg = −0.3 V) applies ≈ 0.25 rad of phase (dwell-time integral
> `∫V dt/ħ` at group velocity ≈ 5.5 × 10⁵ m/s). The two-bin detector therefore
> samples a narrow, near-linear **shoulder of a single lobe**; the scan never
> crosses a fringe maximum or minimum. That — not a special property of the
> device — is why R² ≈ 0.9997 over the shallow window (|Vg| ≤ 0.3 V). The
> deeper-lens tail is an extrapolation of the phase model beyond the
> linear-screening window. The R² is a *regime* statement about where the scan
> sits, not a discovery.

---

## 3b. Coherence budget (stochastic ensemble)

`scripts/ensemble_coherence.py` runs **200 noise realisations** at five
temperatures (4, 10, 20, 50, 77 K) with a delta-correlated dephasing potential
(`entangletronica/stochastic.py`, cell variance `s²/(2 τφ dx dy dt)`), using the
power-law dephasing time τφ(T) = 12 ps·(4/T)^1.5. Results
(`results/coherence_ensemble.json`):

| T (K) | ⟨C⟩ analytical | ⟨C⟩ numerical |
|---|---|---|
| 4 | 0.931 | 0.926 |
| 10 | 0.878 | 0.549 |
| 20 | 0.760 | 0.200 |
| 50 | 0.393 | 0.198 |
| 77 | 0.175 | 0.188 |

The noise amplitude is **empirically calibrated** (s = 22; the raw spec value
2.32 produces no visible dephasing) so that ⟨C⟩(4 K) matches the analytical
anchor. The numerical visibility crosses 0.5 at **T_max ≈ 11 K** — a
conservative operating bound. Above ≈ 20 K the delta-correlated white-noise
model enters its strong-scattering limit and ⟨C⟩ saturates at ≈ 0.19 (a
non-perturbative artefact, not a physical revival); the analytical exponential
is the ideal-device upper envelope. See the paper's methodological note and
`scripts/_calibrate_scale.py`.

**Referee due-diligence checks** (paper §2.4 + §robustness) are committed as
`results/ensemble_convergence.json`, `results/scale_sensitivity.json` and
`results/noise_correlation.json` and re-generated by the scripts below:

- **Noiseless limit** — `stochastic.solve2d_stochastic(scale_noise=0)` is
  bit-identical to the deterministic solver (max |Δψ| = 0, norm conserved to
  1e-12); the full-line contrast of the noiseless profile is C_det = 0.99999
  (the 0.95 prefactor in the analytical envelope is a separate
  single-particle loss convention).
- **Ensemble-size convergence** — nested analysis (same first-N realisations)
  gives C(4 K) = 0.926 at N = 200, flat to ±0.010 for every N ≥ 50 (met);
  the steepest point, 10 K (C = 0.549, bootstrap 1σ = 0.022), carries a
  wider honest band (±0.072 across the ladder) that is reported, not fitted
  away.
- **Noise-scale sensitivity** — T_max(s = 17/22/27) = 15.0/11.4/7.8 K with
  dT_max/ds ≈ −0.73 K per unit: T_max = 11.4 ± 3.6 K for s = 22 ± 5.
- **Spatially correlated noise** — at fixed local amplitude, a finite
  correlation length ξ suppresses the *differential* (between-path) phase
  (common-mode cancellation), so T_max *rises*: 11.4 K (ξ=0) → 14.8 K
  (ξ=5 nm) → 29.2 K (ξ=10 nm). The delta-correlated model is the
  conservative worst case.

---

## 4. Reproduce

Everything — figures, the interference animation and the reported metrics —
regenerates deterministically with this short pipeline (**~12 s** for the
physics part, same output across runs):

```bash
pip install -r requirements.txt
python scripts/make_figures.py           # fig1–5 + metrics (figures/*, results/entangletron_metrics.json)
python scripts/ensemble_coherence.py     # 200×5 stochastic ensemble (~3–4 min, 4-core parallel)
python scripts/ensemble_convergence.py   # nested C(N) check, ±0.01 at 4 K (~2–3 min)
python scripts/scale_sensitivity.py      # T_max(s=22±5) = 11.4±3.6 K (~4–5 min)
python scripts/noise_correlation.py      # ξ = 0/5/10 nm correlated noise (~4 min)
python scripts/make_robustness_figure.py # fig_robustness.pdf multi-panel (sec)
python scripts/make_missing_figures.py   # fig_poisson_mapping, fig_coherence, fig_xor_schematic (paper §2–4)
python scripts/make_animation.py         # assets/iframe_flight.gif
python -m pytest tests/ -q               # full suite (smoke + regression pins + physics tests)
```

Outputs: `figures/*.{pdf,png}`, `results/entangletron_metrics.json`,
`results/coherence_ensemble.json`, `results/ensemble_convergence.json`,
`results/scale_sensitivity.json`, `results/noise_correlation.json`,
`results/*.npy`, `assets/iframe_flight.gif`.

To use `entangletronica` as an importable library (outside the repo scripts),
`pip install -e .` also works; the paper (optional) is built with
`pdflatex paper/EQLI_PhaseGate_Benchmark_2026.tex`.

### Animation — the electron building the interference figure in flight

![Flying electron: interference in flight](assets/iframe_flight.gif)

---

## 5. Conventions

Natural units `ħ = m = 1`; lengths in nm, energies in meV (conversions in
`entangletronica/potential.py`). The grid 140 × 80 @ `dx = 2 nm` resolves `k₀`
with a 4× Nyquist margin.

---

## 6. Related work

This study sits on a well-established experimental literature. Key anchors
(list not exhaustive):

- **Single-electron (edge-channel) Mach–Zehnder interferometer:** Y. Ji,
  Y. Chung, D. Sprinzak, M. Heiblum, D. Mahalu, H. Shtrikman,
  *Nature* **422**, 415 (2003). The canonical experimental electronic
  Mach–Zehnder interferometer in a 2DEG.
- **Electronic Mach–Zehnder coherence / shot noise:** I. Neder et al.,
  *Nature* **448**, 333 (2007); D. Roulleau et al., *PRL* **100**, 126802 (2008).
- **Aharonov–Bohm interference in quantum wires:** e.g., the extensive
  Aharonov–Bohm and quantum-circle literature on interference of single
  electrons in 2DEGs.
- **Gate-defined double-slit and interferometry in 2DEGs:** a long-running
  theme (double-slit gate-defined devices, edge and bulk interferometry
  landscapes developed since the 1990s).

This work is a **reproducible simulation within that tradition** — it borrows
no new physics, and is intended as a compact, well-tested numerical model.

---

## 7. Limitations & caveats

- **No experimental anchoring:** parameters are realistically plausible for an
  InGaAs 2DEG but chosen for a clean simulation, not matched to a published
  device dataset. Any claim of quantitative agreement with a specific
  microphone-oriented experiment would be unfounded.
- **Single-particle, single-slit-splitting physical content:** the result is
  single-electron interference; entanglement (multi-particle) is explicitly out
  of scope, and the name "Entangletronica" does not imply its content.
- **Dephasing modelled as delta-correlated white noise** (paper Sec. 2.4): an
explicit, calibrated model of thermal dephasing, not a microscopic disorder
calculation. No SO coupling or magnetic field. Real devices would add further
channels (e.g. charge noise with finite correlation time) and can only reduce
the clean interference.
- **Numbers labelled "results" are solver self-consistency**, not observed
  phenomena; they justify the calculator's trust, not a scientific claim.

---

<div align="center"><sub>Fabrizio Terzi · MIT License</sub></div>

## Citation (How to cite)

This study is released on Zenodo and identified by the DOI below; please cite
the archived version of record:

> **Fabrizio Terzi**, *Flying-Electron Interference Logic via Electrostatic
> Quantum Lenses (Entangletronica)*, Zenodo, DOI: 10.5281/zenodo.21802416 (2026).
> https://zenodo.org/records/21802416

## Reproducibility & validation (2025-08-02)

- **Dependencies**: install from `requirements-lock.txt` for exact versions
  (matplotlib is tracked as PyPI `3.10.1`, the portable form of the local
  Debian `3.10.1+dfsg1`).
- **Metrics reconciliation**: `results/entangletron_metrics.json` now tags
  each convergence block with its readout functional —
  `box_sum_legacy` (raw grid box-sum) and `continuous_interpolated` (imported
  from `readout_sensitivity.json`). Regenerate with `python scripts/make_figures.py`.
- **Regression guards**: `tests/test_metrics_regression.py` pins the headline
  exports (`transfer_slope_per_kphi`, `transfer_linear_r2`, `max_imbalance`) to
  1e-6 and checks the self-labelling contract; `tests/test_physics.py` guards
  the Poisson–TF mapping (50 meV/V), the stochastic dephasing variance, and the
  coherence ensemble (monotonic C(T), C(4 K) = 0.93 ± 0.05).
- **CI**: `.github/workflows/ci.yml` installs from the lock, rebuilds all
  figures/metrics from the raw simulation, then runs the full test suite.
- **Transit-time note**: the paper Discussion distinguishes the simulated
  window (0.14 ps, source→~103 nm) from the full channel-end transit
  (0.24 ps); see the paper.

---

## Repository structure

```
.
├── paper/
│   ├── EQLI_PhaseGate_Benchmark_2026.tex   # Main LaTeX source (EQLI paper)
│   └── EQLI_PhaseGate_Benchmark_2026.pdf    # Compiled PDF (13 pages)
├── figures/                     # Vector PDFs + PNG previews (14 figures)
├── scripts/
│   ├── entangletron_experiment.py  # Full experiment + figure pipeline
│   ├── make_figures.py             # Figure entry point (single source of truth)
│   ├── ensemble_coherence.py       # 200×5 stochastic dephasing ensemble (JSON + fig)
│   ├── ensemble_convergence.py     # Nested C(N) convergence check (JSON)
│   ├── scale_sensitivity.py        # T_max vs noise scale s = 17/22/27 (JSON)
│   ├── noise_correlation.py        # ξ = 0/5/10 nm correlated-noise check (JSON)
│   ├── make_robustness_figure.py   # 4-panel referee due-diligence figure
│   ├── _calibrate_scale.py         # Empirical scale_noise calibration (dev artifact)
│   ├── make_missing_figures.py     # Poisson-mapping / coherence / XOR figures
│   ├── convergence_study.py        # Grid-refinement study
│   ├── readout_sensitivity.py      # Readout-functional analysis
│   └── make_animation.py           # In-flight interference animation
├── entangletronica/             # Simulation package (potential, electron, gates)
├── results/                     # Simulation outputs (tracked JSON metrics, gitignored *.npy)
├── tests/                       # Smoke + metrics-regression tests
├── assets/                      # Animation artifacts
├── .github/workflows/ci.yml     # CI: clean-install rebuild + tests
├── CITATION.cff                 # Citation metadata (Zenodo/GitHub)
├── CONTRIBUTING.md              # Contribution guide
└── CODE_OF_CONDUCT.md           # Community standards
```

---

## 🔬 Methodological Note

### Multi-LLM Orchestration & Human-in-the-Loop Workflow

- **Multi-Agent Setup**:
  - **Gemini**: Architectural framing, prompt design, and synthesis.
  - **DeepSeek**: Code execution, Schrödinger solver implementation, and LaTeX drafting.
  - **Kimi AI**: Adversarial peer review, edge-case audit, and physical validation.
  - **Pi Agent**: Automated repository maintenance, compilation, and version control.
- **Human Oversight**: Directed by Fabrizio Terzi (Pyragogy.org). The human lead audited the numerical data, caught the dephasing-budget contradiction, recalculated the analytical visibility curve, and required an explicit 200-realisation ensemble plus empirical noise calibration before Table 1 was trusted.
- **Repository**: Publicly tracked on GitHub for open review and replication.

---

## Contributing & community

- **Contribute**: see [`CONTRIBUTING.md`](CONTRIBUTING.md) — ground rules,
  validation suite, and paper-change workflow.
- **Community standards**: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- **License**: [MIT](LICENSE).
- **Citation metadata**: [`CITATION.cff`](CITATION.cff) (Zenodo-ready).
