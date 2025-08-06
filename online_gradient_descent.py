"""Illustrate how online gradient descent handles nonstationary environments."""

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Use continuous time so we can tweak at low FPS then render at high FPS.
RATE = 2.0
PERIODS = np.array([8.0, 1.0])
LAPS = 1
FPS = 12
DT = 1.0 / FPS


class OGDPlot:
    def __init__(self, ax, omega, T):
        self.omega = omega
        self.xs = np.zeros((T, 2))
        self.xopts = np.zeros((T, 2))
        self.xs_plot = ax.plot([], [], label="$y_t$ (ALG)", color="black")[0]
        self.xopts_plot = ax.plot([], [], label="$y^\\star_t$", alpha=0.25, color="black")[0]
        box = 1.1
        ax.set(xlim=[-box, box], ylim=[-box, box])
        ax.axis("equal")
        sns.despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=[], yticks=[])
        self.x = np.zeros(2)

    def step(self, i):
        self.xs_plot.set_data(self.xs[:i, 0], self.xs[:i, 1])
        self.xopts_plot.set_data(self.xopts[:i, 0], self.xopts[:i, 1])
        theta = self.omega * DT * i
        xopt = np.array([np.cos(theta), np.sin(theta)])
        x = self.x
        self.xs[i] = x
        self.xopts[i] = xopt
        grad = x - xopt
        self.x = x - DT * RATE * grad


def main():
    T = int(FPS * LAPS * PERIODS[0]) + 1
    OMEGAS = 2 * np.pi / PERIODS

    xs = np.zeros((2, T, 2))
    xopts = np.zeros((2, T, 2))

    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=200)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.92, wspace=0.25)
    plots = [OGDPlot(ax, omega, T) for ax, omega in zip(axs, OMEGAS)]
    axs[0].set_title("slow-moving target")
    axs[1].set_title("fast-moving target")

    # move legend to in-between
    axs[0].legend()
    legend = axs[0].get_legend_handles_labels()
    fig.legend(*legend, loc='upper center', fontsize="large")
    fig.text(
        0.5, 0.8,
        "$f_t(y_t) = \\|y_t - y^\\star_t\\|_2^2$",
        ha="center", va="top", fontsize="large",
    )
    axs[0].get_legend().remove()

    writer = matplotlib.animation.FFMpegWriter(fps=FPS, bitrate=100*FPS)
    writer.setup(fig, "ogd.mp4")

    for i in range(T):
        for plot in plots:
            plot.step(i)
        writer.grab_frame()
        if FPS < 60:
            plt.show(block=False)
            plt.pause(1e-2)

    writer.finish()


if __name__ == "__main__":
    main()
