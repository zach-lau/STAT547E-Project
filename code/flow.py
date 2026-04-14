"""
Side-by-side comparison:
  Left  — Monge-Kantorovich optimal transport plan (ε → 0 limit)
  Right — Schrödinger Bridge steady state         (ε = 0.05)

Same marginals as sinkhorn.py:
  rows (r): bimodal mixture of Gaussians
  cols (c): right-skewed exponential + bump
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

# ── grid ──────────────────────────────────────────────────────────────────────
N = 50
x = np.linspace(0, 1, N)

# ── target marginals (identical to sinkhorn.py) ───────────────────────────────
def bimodal(x):
    g1 = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
    g2 = np.exp(-0.5 * ((x - 0.72) / 0.12) ** 2)
    p = 0.55 * g1 + 0.45 * g2
    return p / p.sum()

def skewed(x):
    p = np.exp(-4.5 * x) + 0.15 * np.exp(-0.5 * ((x - 0.85) / 0.07) ** 2)
    return p / p.sum()

r = bimodal(x)
c = skewed(x)

# ── Monge-Kantorovich: comonotone coupling ────────────────────────────────────
# For squared-distance cost on R the OT plan is unique and comonotone:
# match mass greedily in order (north-west corner on the sorted marginals).
def monge_kantorovich(r, c):
    N = len(r)
    P = np.zeros((N, N))
    r_rem = r.copy().astype(float)
    c_rem = c.copy().astype(float)
    i = j = 0
    while i < N and j < N:
        t = min(r_rem[i], c_rem[j])
        P[i, j] = t
        r_rem[i] -= t
        c_rem[j] -= t
        if r_rem[i] < 1e-15:
            i += 1
        if c_rem[j] < 1e-15:
            j += 1
    return P

P_mk = monge_kantorovich(r, c)

# ── Schrödinger Bridge: Sinkhorn to convergence ───────────────────────────────
eps = 0.05
K = np.exp(-(x[:, None] - x[None, :]) ** 2 / eps)

u = np.ones(N)
v = np.ones(N)
for _ in range(2000):
    v = c / (K.T @ u)
    u = r / (K @ v)

P_sb = u[:, None] * K * v[None, :]

# ── shared color scale ────────────────────────────────────────────────────────
vmax = max(P_mk.max(), P_sb.max()) * 1.05
vmin = max(vmax * 1e-5, min(P_mk[P_mk > 0].min(), P_sb[P_sb > 0].min()))

CMAP        = "inferno"
BAR_COLOR   = "#4fc3f7"
TARGET_COLOR = "#ff7043"
DARK        = "#1a1a2e"
TEXT        = "#e0e0e0"

# ── draw one panel ─────────────────────────────────────────────────────────────
def draw_panel(fig, outer_ss, P, title):
    """
    Draws a single (heatmap + marginals) panel into the subplot-spec `outer_ss`.
    Returns the three axes for further tweaking.
    """
    inner_v = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer_ss,
        height_ratios=[1, 5, 0.05],
        hspace=0.10,
    )
    inner_h = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=inner_v[1],
        width_ratios=[1, 5],
        wspace=0.06,
    )

    ax_top  = fig.add_subplot(inner_v[0])
    ax_left = fig.add_subplot(inner_h[0])
    ax_main = fig.add_subplot(inner_h[1])

    for ax in [ax_top, ax_left, ax_main]:
        ax.set_facecolor(DARK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=TEXT, labelsize=7)

    # heatmap
    ax_main.imshow(
        P,
        origin="upper",
        aspect="auto",
        cmap=CMAP,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_title(title, color=TEXT, fontsize=12, pad=6, fontweight="bold")

    # column marginals (top)
    idx = np.arange(N)
    col_marg = P.sum(axis=0)
    ax_top.bar(idx, col_marg,  color=BAR_COLOR,    alpha=0.85, width=1.0, label="plan")
    ax_top.step(idx, c,        color=TARGET_COLOR, linewidth=1.4, where="mid", label="target")
    ax_top.set_xlim(-0.5, N - 0.5)
    ax_top.set_ylim(0, max(c.max(), col_marg.max()) * 1.3)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title("Column marginal", color=TEXT, fontsize=8, pad=3)
    ax_top.legend(fontsize=7, framealpha=0.0, labelcolor=TEXT,
                  loc="upper right", handlelength=1.2)

    # row marginals (left, horizontal)
    row_marg = P.sum(axis=1)
    ax_left.barh(idx, row_marg, color=BAR_COLOR,    alpha=0.85, height=1.0)
    ax_left.step(r,   idx,      color=TARGET_COLOR, linewidth=1.4, where="mid")
    ax_left.set_ylim(-0.5, N - 0.5)
    ax_left.set_xlim(0, max(r.max(), row_marg.max()) * 1.4)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.invert_xaxis()
    ax_left.set_title("Row\nmarginal", color=TEXT, fontsize=8, pad=3)

    return ax_top, ax_left, ax_main


# ── build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor=DARK)

outer = gridspec.GridSpec(
    1, 2,
    figure=fig,
    wspace=0.12,
    left=0.03, right=0.97,
    top=0.93,  bottom=0.05,
)

draw_panel(fig, outer[0], P_mk, "Monge–Kantorovich  (ε → 0)")
draw_panel(fig, outer[1], P_sb, f"Schrödinger Bridge  (ε = {eps})")

# shared super-title
fig.suptitle(
    "Optimal Transport  vs  Schrödinger Bridge",
    color=TEXT, fontsize=14, fontweight="bold", y=0.99,
)

out_path = "/Users/zacharylau/Desktop/School/2025-26/Sampling/project/ot_vs_sb.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {out_path}")
