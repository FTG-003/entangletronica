# Contributing to Entangletronica / EQLI

Thanks for your interest! This is a small, reproducible numerical study; the
maintainer keeps the bar high on **reproducibility** and **honest framing**.
Before opening a PR, please read this page — and if in doubt, open an issue
first.

## Ground rules

1. **No claims inflation.** This is a *single-particle* simulation study.
   PRs must not add claims of new physics, entanglement, or experimental
   verification. The README's "honest meaning" table is the contract.
2. **Reproducibility is a feature.** Any change that alters numbers or figures
   must keep the pipeline deterministic and update the affected artifacts
   (scripts, regression tests, figures).
3. **Small, focused PRs.** One logical change per PR. No drive-by refactors.

## How to contribute

### Bug reports & feature requests

Open an issue with a minimal reproduction. For numeric issues, include:
- the exact command run and Python version;
- `requirements-lock.txt` or `pip freeze`;
- expected vs observed output.

### Pull requests

1. Fork the repository.
2. Create a branch: `git checkout -b fix/describe-the-fix`.
3. Make your change. Follow the existing style (PEP 8, docstrings, natural
   units `ħ = m = 1` documented in `entangletronica/potential.py`).
4. Run the validation suite:
   ```bash
   python tests/test_smoke.py
   python tests/test_metrics_regression.py
   python scripts/make_figures.py    # if figures/metrics change
   ```
   The CI workflow (`.github/workflows/ci.yml`) runs the same pipeline from a
   clean install of `requirements-lock.txt`.
5. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   message, e.g. `fix(readout): interpolate detector profile before binning`.
6. Push and open the PR against `main`. Keep the PR description explicit about
   what changed and why.

### LaTeX / paper changes

If you touch `paper/EQLI_PhaseGate_Benchmark_2026.tex`:
- rebuild with `pdflatex` (3 passes) and check the log is clean
  (no undefined references, no overfull boxes > 3 pt);
- verify every new `\cite` has a `\bibitem` and every DOI resolves;
- add the regenerated `paper/EQLI_PhaseGate_Benchmark_2026.pdf` to the PR.

## Style

- Python: PEP 8; type hints welcome but not required; `numpy`-first idioms.
- Docstrings: one-line summary + units for any quantity.
- Figures: vector PDFs via matplotlib, `bbox_inches="tight"`, fonts sized for
  single-column reproduction.
- Git: Conventional Commits; squash history on merge.

## Code of conduct

All interactions are governed by `CODE_OF_CONDUCT.md`. Be kind; review the
science, not the person.
