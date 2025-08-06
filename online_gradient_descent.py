"""Illustrate how online gradient descent handles nonstationary environments."""

import matplotlib.animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Use continuous time so we can tweak at low FPS then render at high FPS.
RATE = 2.0
PERIODS = np.array([6.0, 1.0])
LAPS = 4
FPS = 12
DT = 1.0 / FPS
BUF_SEC = 1.0
BUF = int(BUF_SEC / DT)

COLOR_X = 0.0
COLOR_XOPT = 0.7


class DecayingTrace:
    def __init__(self, ax, N, base_color, **kwargs):
        self.segments = np.zeros((N, 2, 2))
        self.segments[:, 1, 1] = 1
        bc = base_color
        c0 = [bc, bc, bc, 1]
        c1 = [bc, bc, bc, 0]
        cmap = LinearSegmentedColormap.from_list("", [c0, c1])
        self.cursor = 0
        self.lc = LineCollection(self.segments, cmap=cmap, **kwargs)
        self.lc.set_array(np.linspace(0, 1, N))
        self.init = False
        self.lc = ax.add_collection(self.lc)

    def step(self, x):
        if not self.init:
            self.segments[:, :, :] = x[None, None, :]
            self.init = True
        else:
            self.segments = np.roll(self.segments, 1, axis=0)
            self.segments[0, 0, :] = self.segments[1, 1, :]
            self.segments[0, 1, :] = x
        self.lc.set_segments(self.segments)
        self.lc.changed()


class OGDPlot:
    def __init__(self, ax, omega):
        self.omega = omega
        self.xtrace = DecayingTrace(ax, BUF, COLOR_X, label="$y_t$ (ALG)")
        self.xopttrace = DecayingTrace(ax, BUF, COLOR_XOPT, label="$y^\\star_t$")

        self.x_plot = ax.plot([], [], marker=".", markersize=10, color="black")[0]
        self.xopt_plot = ax.plot([], [], marker=".", markersize=10, color="gray")[0]

        ax.axis("equal")
        sns.despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=[], yticks=[])
        self.x = np.zeros(2)

        self.ax = ax

    def step(self, i):
        theta = self.omega * DT * i
        xopt = np.array([np.cos(theta), np.sin(theta)])
        grad = self.x - xopt
        self.x = self.x - DT * RATE * grad
        print(self.x)
        print(xopt)
        self.x_plot.set_data([self.x[0]], [self.x[1]])
        self.xopt_plot.set_data([xopt[0]], [xopt[1]])
        self.xtrace.step(self.x)
        self.xopttrace.step(xopt)
        box = 1.1
        self.ax.set(xlim=[-box, box], ylim=[-box, box])


def main():
    T = int(FPS * LAPS * PERIODS[0]) + 1
    OMEGAS = 2 * np.pi / PERIODS

    xs = np.zeros((2, T, 2))
    xopts = np.zeros((2, T, 2))

    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=200)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.92, wspace=0.25)
    plots = [OGDPlot(ax, omega) for ax, omega in zip(axs, OMEGAS)]
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
