# Diffusion Schrödinger Bridges
README by Claude

STAT 547E project exploring Schrödinger bridges — a classical physics problem with modern applications to generative modeling and diffusion models.

## Contents

- `report/` — LaTeX report (`zach_report.tex`)
- `presentation/` — Quarto/reveal.js slides
- `code/` — Python visualizations
- `graphics/` — Generated figures and animations

## Setup

```bash
source .venv/bin/activate   # Python 3.14.3 (numpy, scipy, matplotlib, pillow)
```

Julia 1.11.9 is also configured via `Project.toml`.

## Running the Code

```bash
python code/annealing_vs_sb.py    # → graphics/annealing_vs_sb.png
python code/flow.py               # → graphics/ot_vs_sb.png, sb.png, OT.png, independent.png
python code/brownian_bridge.py    # → graphics/brownian_bridge.gif (~100MB)
python code/sinkhorn.py           # → graphics/sinkhorn.gif
```

The interactive particle simulation is a standalone HTML file: `code/particle_sim.html`.
