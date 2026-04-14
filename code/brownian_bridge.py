import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(42)

# Parameters
N = 500          # number of time steps
M = 100          # number of particles
t0, t1 = 0, 1   # start and end times
x0, x1 = 0, 0   # start and end positions

# Generate M Brownian bridges simultaneously (shape: M x N+1)
dt = (t1 - t0) / N
t = np.linspace(t0, t1, N + 1)

increments = np.random.normal(0, np.sqrt(dt), (M, N))
W = np.hstack([np.zeros((M, 1)), np.cumsum(increments, axis=1)])

# Brownian bridge: pin both endpoints
drift = x0 + (x1 - x0) * (t - t0) / (t1 - t0)          # shape: (N+1,)
bridges = drift + (W - ((t - t0) / (t1 - t0)) * W[:, -1:])  # shape: (M, N+1)

# --- Animation ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(t0, t1)
ypad = max(0.5, np.abs(bridges).max() * 0.15)
ax.set_ylim(bridges.min() - ypad, bridges.max() + ypad)
ax.set_xlabel("t")
ax.set_ylabel("X(t)")
ax.set_title(f"Brownian Bridge — {M} particles")

# Draw all full paths faintly in the background
for path in bridges:
    ax.plot(t, path, color="steelblue", lw=0.4, alpha=0.15, zorder=1)

# Mark fixed endpoints
ax.plot([t0, t1], [x0, x1], "o", color="black", ms=6, zorder=3)

# One trail + dot per particle
trails = [ax.plot([], [], color="steelblue", lw=0.8, alpha=0.4, zorder=2)[0] for _ in range(M)]
dots   = ax.plot([], [], "o", color="crimson", ms=3, alpha=0.6, zorder=4, linestyle="None")[0]

# Subsample frames so the GIF isn't enormous
frame_indices = np.linspace(0, N, 200, dtype=int)

def init():
    for trail in trails:
        trail.set_data([], [])
    dots.set_data([], [])
    return trails + [dots]

def update(frame_idx):
    i = frame_indices[frame_idx]
    for k, trail in enumerate(trails):
        trail.set_data(t[: i + 1], bridges[k, : i + 1])
    dots.set_data(t[i] * np.ones(M), bridges[:, i])
    return trails + [dots]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    init_func=init,
    interval=30,
    blit=True,
)

output_path = "brownian_bridge.gif"
ani.save(output_path, writer="pillow", fps=33)
print(f"Saved to {output_path}")
plt.close(fig)
