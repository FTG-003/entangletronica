<div align="center">

# Flying-Electron Interference Logic via Electrostatic Quantum Lenses

### Entangletronica

**Fabrizio Terzi** · [Entangletronica Lab](https://github.com/FTG-003)

*Single-electron ("flying-qubit") interference logic engineered with electrostatic
quantum lenses — reproducible split-step Schrödinger simulations.*

---

**License** [MIT](./LICENSE) · **Status** reproducible · **Solver** split-step FFT

</div>

---

## Abstract

We report a full numerical study of **Entangletronica**, a device architecture in
which a *single flying electron* — not a stationary qubit — is shaped by
electrostatic lenses to implement **interference logic**.

A Gaussian wave packet (`k₀ = 0.20 nm⁻¹`, `E = 36.3 meV`, `λ = 31.4 nm`) is
launched into a ballistic InGaAs 2DEG and split by a **gate-defined double slit**
into two coherent sources. Downstream, a **programmable phase lens**
(`a = −15 meV`) behind the upper aperture imparts a phase shift *in flight*,
`φ(kφ) = −∫V dt/ħ`. The displaced interference figure is read by a two-bin
detector: **gate voltage in, interference-logic out**.

The gate behaves as a **linear transducer**, resolved numerically to
**R² = 0.99997** with a **sensitivity of 0.142 per gate unit**, a maximum
detector imbalance of **0.44**, and **fringe contrast ≥ 0.9** across the scan.
Because the carrier is *flying*, the lens can be modulated during flight —
enabling **programmable in-flight switching on a picosecond scale** — while the
split-step solver preserves exact unitarity (norm conserved to **1e⁻¹³**).

### Animation — the electron building the interference figure in flight

![Flying electron: interference in flight](./assets/iframe_flight.gif)

---

## Results at a glance

| Quantity | Value |
|---|---|
| Solver | Split-step FFT — exact unitarity (norm to **1e⁻¹³**) |
| Transfer characteristic | linear — **R² = 0.99997** |
| Sensitivity | 0.142 per gate unit |
| Max detector imbalance | **0.44** |
| Fringe contrast | **≥ 0.9** across the full scan |
| Energy / wavelength | 36.3 meV / 31.4 nm |
| Switching scale | ~ ps (ballistic transit) |

## Reproduce

All figures, the interference animation and the reported metrics regenerate
from a single command (**~12 s**):

```bash
pip install -r requirements.txt
python scripts/make_figures.py     # figures + metrics (R² = 0.99997, imbalance…)
python scripts/make_animation.py   # assets/iframe_flight.gif
python tests/test_smoke.py         # unitarity + architecture sanity checks
```

Outputs land in `figures/*.{pdf,png}`, `results/entangletron_metrics.json`,
`results/*.npy`, and `assets/iframe_flight.gif`.

## Publication

The companion paper — *Flying-Electron Interference Logic via Electrostatic
Quantum Lenses (Entangletronica)* — is compiled from `paper/paper.tex`
(`paper/paper.pdf`, **5 pages**, **pdflatex**).

## Layout

```
entangletronica/   core module (potential, electron, gates, simulate)
scripts/           pipeline: experimentation, figures, animation
figures/           generated figures + animation frames
assets/            public multimedia (iframe_flight.gif)
results/           cached metrics and arrays
paper/             LaTeX publication
tests/             sanity checks
```

## Conventions

Natural units `hbar = m = 1`; lengths in nm, energies in meV (see
`entangletronica/potential.py` for conversions). Grid: 140×80, `dx = 2 nm`
(resolves `k₀` with a **4×** Nyquist margin — the grid that makes the physics
correct, cf. `NOTES.md`).

<div align="center"><sub>© Fabrizio Terzi · MIT License</sub></div>