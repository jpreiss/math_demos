"""Show how the infinity-norm ball grows relative to the 2-norm ball.

In n-dimensional Euclidean space, the vector (1, 0, ..., 0) has Euclidean norm
1 infinity-norm 1. On the other hand, the vector (1, ..., 1) still has
infinity-norm 1, but Euclidean norm sqrt(n). The measure of the unit ball as a
fraction of the unit cube goes to zero as n goes to infinity. This is at the
root of many "curse of dimensionality" issues with computations in Euclidean
space.

Here, we illustrate the phenomenon by imagining the following process:
- Start at the vector (1, 0, ..., 0).
- Take a step towards the vector (1, ..., 1).
  Now our position is (1, x, ..., x) for some 0 < x < 1.
- Measure the Euclidean norm of our position.
- Plot a point that is on the same ray as (1, x), but with our length.
  Note that when n=2, this means we are plotting (1, x) exactly, so we are
  plotting the 2-dimensional infinity-norm ball.
- Take another step, and repeat until x = 1.
  Then rotate and reflect our path to form a closed curve in 2D.
"""

import matplotlib.pyplot as plt
import numpy as np


def rotplot(xy, **kwargs):
    reflect = np.array([[0, 1], [1, 0]])
    rotate = np.array([[0, 1], [-1, 0]])
    x = []
    y = []
    for _ in range(8):
        x.extend(xy[0][:-1])
        y.extend(xy[1][:-1])
        xy = (reflect @ xy)[:,::-1]
        reflect = reflect @ rotate
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, **kwargs)


def main():
    dimensions = 2 ** np.arange(1, 5)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(dimensions) + 1))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), tight_layout=True)

    # Plot the Euclidean circle for reference.
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), color=colors[0], label=f"euclidan")

    # Sweep the value of the 2nd, ..., nth entries of the n-dimensional vector.
    steps = np.linspace(0, 1, 100)

    for n, color in zip(dimensions, colors[1:]):
        x = np.tile(steps, (n, 1))
        x[0, :] = 1.0
        norm_ratios = np.linalg.norm(x, axis=0) / np.linalg.norm(x[:2], axis=0)
        x_plot = norm_ratios * x[:2]
        rotplot(x_plot, label=f"inf, d={n}", color=color, solid_joinstyle="miter")

    # Make the plot look nice.
    ax.axis("equal")
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(direction="inout", length=10)
    fig.legend(loc="right")
    fig.savefig("norm_ratios.pdf")


if __name__ == "__main__":
    main()
