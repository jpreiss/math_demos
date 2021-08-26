"""Visualizes different behaviors of 2-dimensional linear dynamical systems."""

import itertools as it
import numpy as np
import matplotlib.pyplot as plt


def rollout(A, x, T, dt):
    """rolls out linear dynamicx x' = Ax for T steps with time interval dt."""
    out = np.zeros((T,) + x.shape)
    out[0] = x
    for t in range(1, T):
        out[t] = out[t-1] + dt * A @ out[t-1]
    return out


def trdet(tr, det):
    """Constructs a 2x2 matrix with the given trace and determinant."""
    a = d = tr / 2
    bc = a*d - det
    b = c = np.sqrt(np.abs(bc))
    if bc < 0:
        b = -b
    return np.array([[a, b], [c, d]])


def main():
    tiles = 7
    points_per_tile = 200
    integrate_steps = 400
    integrate_dt = 0.01

    traces = np.linspace(-1, 1, tiles)
    determinants = np.linspace(-1, 1, tiles)
    halfbox = (traces[1] - traces[0]) / 2.0

    plt.figure(figsize=(10, 10), constrained_layout=True)

    for tr, det in it.product(traces, determinants):
        # Create matrix and evaluate dynamics from random initial points.
        A = trdet(tr, det)
        x0 = np.random.uniform(-halfbox, halfbox, size=(2, points_per_tile))
        x = rollout(A, x0, integrate_steps, integrate_dt)

        # Cut off points that escaped this tile's box.
        for i in range(points_per_tile):
            escaped = np.any(np.abs(x[:, :, i]) > halfbox, axis=1)
            x[escaped, :, i] = np.nan

        # Plot many lines at once.
        color = (0, 0, 0)
        plt.plot(x[:, 0, :] + tr, x[:, 1, :] + det, color=color, linewidth=0.3)

    plt.box(False)
    plt.axis("equal")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig("phase_portraits.pdf")


if __name__ == "__main__":
    main()

