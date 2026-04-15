"""
Linear Annealing vs Schrödinger Bridge: density interpolation paths.

(a) Unimodal N(0,1)  →  Bimodal mixture of Gaussians
(b) Broad prior      →  Posterior with asymmetric likelihood

For the Schrödinger bridge the coupling P_ε is obtained via Sinkhorn
(kernel K = exp(-|x-y|²/ε)).  The marginal at time t is then:

    p_t(x) = Σ_{i,j} P[i,j] · N(x ; (1-t)x_i + t x_j, t(1-t)ε)

which averages Brownian-bridge densities over the coupling.

Linear annealing uses the geometric interpolation:
    p_t(x) ∝ p_0(x)^(1-t) · p_1(x)^t
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ── grid ──────────────────────────────────────────────────────────────────────
N  = 100
x  = np.linspace(-7, 7, N)
dx = x[1] - x[0]

def normalize(p):
    return p / (p.sum() * dx)

# ── distributions ─────────────────────────────────────────────────────────────

# (a) unimodal N(0,1)
p0_a = normalize(np.exp(-0.5 * x**2))

# (a) bimodal mixture – modes at ±2.5, std 0.7
p1_a = normalize(
    0.5 * np.exp(-0.5 * ((x - 2.5) / 0.7)**2) +
    0.5 * np.exp(-0.5 * ((x + 2.5) / 0.7)**2)
)

# (b) broad symmetric prior
prior = normalize(np.exp(-0.5 * (x / 2.5)**2))

# (b) asymmetric likelihood: sharp primary peak at +3.5, secondary shoulder at +1
lik = np.exp(-0.5 * ((x - 3.5) / 0.45)**2) + 0.45 * np.exp(-0.5 * ((x - 1.0) / 1.1)**2)
lik = np.maximum(lik, 0)

# (b) posterior ∝ prior × likelihood
p0_b = prior.copy()
p1_b = normalize(prior * lik)

# ── linear annealing (geometric path) ─────────────────────────────────────────

def geometric_path(p0, p1, t):
    if t <= 0.0:
        return p0.copy()
    if t >= 1.0:
        return p1.copy()
    lp = (1 - t) * np.log(np.maximum(p0, 1e-300)) + t * np.log(np.maximum(p1, 1e-300))
    lp -= lp.max()
    return normalize(np.exp(lp))

# ── Schrödinger bridge ────────────────────────────────────────────────────────

EPS = 1.2   # diffusion parameter for reference Brownian motion

def sinkhorn_plan(p0, p1, eps=EPS, n_iter=3000):
    K      = np.exp(-(x[:, None] - x[None, :]) ** 2 / eps)
    p0_mass = p0 * dx
    p1_mass = p1 * dx
    u = np.ones(N)
    v = np.ones(N)
    for _ in range(n_iter):
        v = p1_mass / (K.T @ u)
        u = p0_mass / (K    @ v)
    return u[:, None] * K * v[None, :]   # shape (N, N), mass matrix

def sb_marginal(P, t, p0, p1, eps=EPS):
    """Marginal of the SB at time t via Brownian-bridge averaging."""
    if t <= 0.0:
        return p0.copy()
    if t >= 1.0:
        return p1.copy()
    std   = np.sqrt(eps * t * (1 - t))
    means = (1 - t) * x[:, None] + t * x[None, :]        # (N, N)
    # diffs[k, i, j] = x[k] - means[i, j]  →  shape (N, N, N)
    diffs  = x[:, None, None] - means[None, :, :]
    gauss  = np.exp(-0.5 * (diffs / std) ** 2) / (std * np.sqrt(2 * np.pi))
    p_t    = (P[None, :, :] * gauss).sum(axis=(1, 2))
    return normalize(p_t)

# ── compute plans ─────────────────────────────────────────────────────────────

print("Sinkhorn (a)…", flush=True)
P_a = sinkhorn_plan(p0_a, p1_a)
print("Sinkhorn (b)…", flush=True)
P_b = sinkhorn_plan(p0_b, p1_b)
print("Plans ready.", flush=True)

# ── time steps & colour scheme ────────────────────────────────────────────────

times    = [0.0, 0.25, 0.5, 0.75, 1.0]
cmap_name = "plasma"
cmap      = plt.cm.get_cmap(cmap_name)
t_vals    = [0.08, 0.28, 0.52, 0.72, 0.95]   # positions along cmap for visual contrast
t_colors  = [cmap(v) for v in t_vals]
t_lws     = [2.4, 1.8, 1.8, 1.8, 2.4]
t_alphas  = [1.0, 0.85, 0.85, 0.85, 1.0]

# ── style constants ───────────────────────────────────────────────────────────

DARK   = "#0d0d1a"
PANEL  = "#12122a"
EDGE   = "#2e2e50"
TEXT   = "#e8e8f2"

# ── figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 8.5), facecolor=DARK)
gs  = GridSpec(
    2, 2, figure=fig,
    hspace=0.52, wspace=0.28,
    left=0.07, right=0.97, top=0.88, bottom=0.13,
)
axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=8, length=3, width=0.7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

def plot_panel(ax, p0, p1, P, mode, title, row_label=""):
    style_ax(ax)
    for t, col, lw, alpha in zip(times, t_colors, t_lws, t_alphas):
        if mode == "linear":
            pt = geometric_path(p0, p1, t)
        else:
            pt = sb_marginal(P, t, p0, p1)
        ax.plot(x, pt, color=col, lw=lw, alpha=alpha)

    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_yticks([])
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    if row_label:
        ax.text(
            -0.18, 0.5, row_label,
            transform=ax.transAxes,
            color=TEXT, fontsize=11, fontweight="bold",
            ha="center", va="center", rotation=90,
        )

# row (a)
plot_panel(axes[0][0], p0_a, p1_a, P_a, "linear",
           "Linear Annealing", row_label="(a) Unimodal → Bimodal")
plot_panel(axes[0][1], p0_a, p1_a, P_a, "sb",
           "Schrödinger Bridge")

# row (b)
plot_panel(axes[1][0], p0_b, p1_b, P_b, "linear",
           "Linear Annealing", row_label="(b) Prior → Posterior")
plot_panel(axes[1][1], p0_b, p1_b, P_b, "sb",
           "Schrödinger Bridge")

# ── shared time legend / colorbar ─────────────────────────────────────────────

sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cax = fig.add_axes([0.22, 0.04, 0.56, 0.022])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("time  t", color=TEXT, fontsize=10, labelpad=4)
cbar.ax.xaxis.set_tick_params(color=TEXT, labelsize=9)
plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color=TEXT)
# markers at each sampled time
for t, col in zip(times, t_colors):
    cax.axvline(t, color=col, lw=1.8, alpha=0.9)

# ── super-title ───────────────────────────────────────────────────────────────

fig.suptitle(
    "Linear Annealing  vs  Schrödinger Bridge",
    color=TEXT, fontsize=14, fontweight="bold", y=0.96,
)

out = "/Users/zacharylau/Desktop/School/2025-26/Sampling/project/annealing_vs_sb.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {out}")
