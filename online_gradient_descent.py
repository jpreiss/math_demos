"""Illustrate how online gradient descent handles nonstationary environments.
"""

import sys

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Use continuous time so we can tweak at low FPS then render at high FPS.
RATE = 1.0
PERIOD = 6.0
LAPS = 2
FPS = 4
DT = 1.0 / FPS


def main():
    T = int(FPS * LAPS * PERIOD)
    OMEGA = 2 * np.pi / PERIOD
    x = np.zeros(2)

    xs = np.zeros((T, 2))
    xopts = np.zeros((T, 2))

    plt.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=200)
    opt_trace, = ax.plot([], [], label="$y^\\star_t$", alpha=0.25, color="black")
    x_trace, = ax.plot([], [], label="$y_t$ (ALG)", color="black")
    ax.legend()
    ax.set_title("$h_t(y) = \\|y - y^\\star_t\\|_2^2$")
    box = 1.5
    ax.set(xlim=[-box, box], ylim=[-box, box])
    ax.axis("equal")
    sns.despine(ax=ax, bottom=True, left=True)
    ax.set(xticks=[], yticks=[])

    writer = matplotlib.animation.FFMpegWriter(fps=FPS, bitrate=100*FPS)
    writer.setup(fig, "ogd.mp4")

    for i in range(T):
        x_trace.set_data(xs[:i, 0], xs[:i, 1])
        opt_trace.set_data(xopts[:i, 0], xopts[:i, 1])
        writer.grab_frame()
        plt.show(block=False)
        plt.pause(1e-2)

        theta = OMEGA * DT * i
        xopt = np.array([np.cos(theta), np.sin(theta)])
        xs[i] = x
        xopts[i] = xopt
        grad = x - xopt
        x = x - DT * RATE * grad

    writer.finish()


if __name__ == "__main__":
    main()
