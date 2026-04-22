# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project (STAT 547E) on **Diffusion Schrödinger Bridges** — a classical physics problem with modern applications to generative modeling and diffusion models. The project produces Python visualizations, a LaTeX report, and a Quarto/reveal.js presentation.

## Environment

Use the Python environment in `.venv`:

```bash
source .venv/bin/activate
```

Python 3.14.3. No requirements.txt; installed packages include numpy, scipy, matplotlib, and pillow.

Julia 1.11.9 is also configured via `Project.toml` (dependencies: CSV, DataFrames, Distributions, Plots, LinearAlgebra).

## Running Code

All Python scripts in `code/` write their outputs to `graphics/`:

```bash
python code/annealing_vs_sb.py   # → graphics/annealing_vs_sb.png
python code/flow.py              # → graphics/ot_vs_sb.png, sb.png, OT.png, independent.png
python code/brownian_bridge.py   # → graphics/brownian_bridge.gif (~100MB)
python code/sinkhorn.py          # → graphics/sinkhorn.gif
```

The interactive particle simulation is a standalone HTML file: `code/particle_sim.html`.

## Building the Report and Presentation

```bash
# LaTeX report
cd report && latexmk -pdf zach_report.tex

# Quarto presentation (outputs HTML and PDF)
cd presentation && quarto render presentation.qmd
```

## Code Architecture

Each script in `code/` is self-contained and produces a single output artifact:

- **`brownian_bridge.py`** — Simulates 100 particles following Brownian bridge paths between two marginals; produces animated GIF using matplotlib's `FuncAnimation`.
- **`sinkhorn.py`** — Implements the Sinkhorn algorithm for entropy-regularized optimal transport; animates convergence frame-by-frame.
- **`flow.py`** — Compares Monge-Kantorovich (deterministic) optimal transport with the Schrödinger Bridge solution side-by-side.
- **`annealing_vs_sb.py`** — Contrasts linear density annealing (geometric interpolation) against Schrödinger Bridge marginals for two example distributions.

The scripts share a common pattern: define source/target distributions, run an algorithm, then render frames or a static figure via matplotlib.

## Report Structure

`report/zach_report.tex` covers: Brownian bridges → mixtures of Brownian bridges → finite particle perspective on Schrödinger bridges → connection to diffusion models (DDPM, score-based models) → Sinkhorn algorithm → generative modeling formulations (De Bortoli et al.).

Reference PDFs are in the project root (e.g., `ddpm_2006.11239v2.pdf`, `schroding_bridge_2106.01357v5.pdf`).
