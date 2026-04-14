"""
Sinkhorn algorithm visualized as an animated GIF.

Joint distribution: 50x50 matrix.
Row marginal (r): mixture of two Gaussians.
Col marginal (c): right-skewed distribution (exponentially weighted).
Cost: squared Euclidean distance on [0,1]^2.
Regularisation: epsilon = 0.05.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from PIL import Image
import io

# ── grid ──────────────────────────────────────────────────────────────────────
N = 50
x = np.linspace(0, 1, N)

# ── target marginals ──────────────────────────────────────────────────────────
def bimodal(x):
    """Mixture of two Gaussians."""
    g1 = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
    g2 = np.exp(-0.5 * ((x - 0.72) / 0.12) ** 2)
    p = 0.55 * g1 + 0.45 * g2
    return p / p.sum()

def skewed(x):
    """Right-skewed: exponential decay from left."""
    p = np.exp(-4.5 * x) + 0.15 * np.exp(-0.5 * ((x - 0.85) / 0.07) ** 2)
    return p / p.sum()

r = bimodal(x)   # row marginal  (axis-0 / y-axis in imshow)
c = skewed(x)    # col marginal  (axis-1 / x-axis in imshow)

# ── cost matrix & Gibbs kernel ────────────────────────────────────────────────
eps = 0.05
C = (x[:, None] - x[None, :]) ** 2          # shape (N, N)
K = np.exp(-C / eps)                         # Gibbs kernel

# ── Sinkhorn iterations ───────────────────────────────────────────────────────
n_iter = 40
frames_at = list(range(0, n_iter + 1))       # capture every iteration

u = np.ones(N)
v = np.ones(N)

def joint(u, v):
    return u[:, None] * K * v[None, :]

snapshots = []   # list of (iter, P, row_marginal, col_marginal)

for i in range(n_iter + 1):
    P = joint(u, v)
    snapshots.append((i, P.copy(), P.sum(axis=1).copy(), P.sum(axis=0).copy()))
    if i < n_iter:
        v = c / (K.T @ u)
        u = r / (K   @ v)

# ── plotting helpers ──────────────────────────────────────────────────────────
CMAP = "inferno"
vmin, vmax = 1e-6, joint(u, v).max() * 1.05

BAR_COLOR   = "#4fc3f7"
TARGET_COLOR = "#ff7043"

def render_frame(iter_num, P, row_marg, col_marg):
    fig = plt.figure(figsize=(7, 7), facecolor="#1a1a2e")

    # GridSpec: [top-bar | main+left | bottom-pad]
    #           left-strip | main heatmap
    outer = gridspec.GridSpec(
        3, 1,
        height_ratios=[1, 6, 0.15],
        hspace=0.08,
        figure=fig,
    )
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[1],
        width_ratios=[1, 6],
        wspace=0.06,
    )

    ax_top  = fig.add_subplot(outer[0])   # column marginals (top bar chart)
    ax_left = fig.add_subplot(inner[0])   # row marginals    (left bar chart)
    ax_main = fig.add_subplot(inner[1])   # joint heatmap

    dark_bg = "#1a1a2e"
    text_col = "#e0e0e0"

    for ax in [ax_top, ax_left, ax_main]:
        ax.set_facecolor(dark_bg)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=text_col, labelsize=7)

    # ── heatmap ───────────────────────────────────────────────────────────────
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
    ax_main.set_title(
        f"Iteration {iter_num:02d}",
        color=text_col, fontsize=11, pad=4,
    )

    # ── column marginals (top) ─────────────────────────────────────────────
    idx = np.arange(N)
    ax_top.bar(idx, col_marg,  color=BAR_COLOR,    alpha=0.85, width=1.0, label="current")
    ax_top.step(idx, c,        color=TARGET_COLOR, linewidth=1.4, where="mid", label="target")
    ax_top.set_xlim(-0.5, N - 0.5)
    ax_top.set_ylim(0, max(c.max(), col_marg.max()) * 1.25)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title("Column marginal  (current vs target)", color=text_col, fontsize=8, pad=3)
    ax_top.legend(
        fontsize=7, framealpha=0.0,
        labelcolor=text_col, loc="upper right",
        handlelength=1.2,
    )

    # ── row marginals (left, horizontal bars) ─────────────────────────────
    ax_left.barh(idx, row_marg, color=BAR_COLOR,    alpha=0.85, height=1.0)
    ax_left.step(r,   idx,      color=TARGET_COLOR, linewidth=1.4, where="mid")
    ax_left.set_ylim(-0.5, N - 0.5)
    ax_left.set_xlim(0, max(r.max(), row_marg.max()) * 1.35)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.invert_xaxis()
    ax_left.set_title("Row\nmarginal", color=text_col, fontsize=8, pad=3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=dark_bg)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── render & save ─────────────────────────────────────────────────────────────
print("Rendering frames …")
images = []
for iter_num, P, row_marg, col_marg in snapshots:
    img = render_frame(iter_num, P, row_marg, col_marg)
    images.append(img)
    print(f"  frame {iter_num:02d}/{n_iter}")

out_path = "/Users/zacharylau/Desktop/School/2025-26/Sampling/project/sinkhorn.gif"
images[0].save(
    out_path,
    save_all=True,
    append_images=images[1:],
    duration=500,     # 500 ms per frame
    loop=0,
)
print(f"\nSaved → {out_path}")
