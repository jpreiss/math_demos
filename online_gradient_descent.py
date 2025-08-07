"""Illustrate how online gradient descent handles nonstationary environments."""

import matplotlib.animation
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Use continuous time so we can tweak at low FPS then render at high FPS.

# cts-time setup
RATE = 2.0
PERIODS = np.array([6.0, 1.0])
LAPS = 2
BUF_SEC = 1.5
COLOR_X = [0, 0, 0, 1]
COLOR_XOPT = [0.7, 0.7, 0.7, 1]

# discretization
SCENE_FPS = 240
SKIP = 16
DT = 1.0 / SCENE_FPS
VIDEO_FPS = SCENE_FPS / SKIP
BUF = int(BUF_SEC / DT)


class DecayingTrace:
    def __init__(self, ax, N, base_color, **kwargs):
        self.segments = np.zeros((N, 2, 2))
        c1 = base_color[:3] + [0]
        cmap = LinearSegmentedColormap.from_list("", [base_color, c1])
        self.lc = LineCollection(self.segments, cmap=cmap, **kwargs)
        self.lc.set_array(np.linspace(0, 1, N))
        self.init = False
        self.lc = ax.add_collection(self.lc)

    def step(self, x, render):
        if not self.init:
            self.segments[:, :, :] = x[None, None, :]
            self.init = True
        else:
            self.segments = np.roll(self.segments, 1, axis=0)
            self.segments[0, 0, :] = self.segments[1, 1, :]
            self.segments[0, 1, :] = x
        if render:
            self.lc.set_segments(self.segments)
            self.lc.changed()


class OGDPlot:
    def __init__(self, ax, omega, buf):
        self.omega = omega
        self.xtrace = DecayingTrace(ax, buf, COLOR_X)
        self.xopttrace = DecayingTrace(ax, buf, COLOR_XOPT)

        self.x_plot = ax.plot([], [], marker=".", markersize=10, color=COLOR_X)[0]
        self.xopt_plot = ax.plot([], [], marker=".", markersize=10, color=COLOR_XOPT)[0]

        ax.axis("equal")
        sns.despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=[], yticks=[])
        self.x = np.zeros(2)

        self.ax = ax

    def step(self, i, render):
        theta = self.omega * DT * i
        xopt = np.array([np.cos(theta), np.sin(theta)])
        grad = self.x - xopt
        self.x = self.x - DT * RATE * grad
        self.x_plot.set_data([self.x[0]], [self.x[1]])
        self.xopt_plot.set_data([xopt[0]], [xopt[1]])
        self.xtrace.step(self.x, render)
        self.xopttrace.step(xopt, render)
        box = 1.1
        self.ax.set(xlim=[-box, box], ylim=[-box, box])


def main():
    T = int(SCENE_FPS * LAPS * PERIODS[0]) + 1
    OMEGAS = 2 * np.pi / PERIODS

    xs = np.zeros((2, T, 2))
    xopts = np.zeros((2, T, 2))

    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=200)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.92, wspace=0.25)
    buflens = [BUF, BUF // 2]
    plots = [
        OGDPlot(ax, omega, buf)
        for ax, omega, buf in zip(axs, OMEGAS, buflens)
    ]
    axs[0].set_title("slow-moving target")
    axs[1].set_title("fast-moving target")

    # legend in-between, and we need to rebuild the lines because the
    # colormapped LineCollection legends don't work.
    handles = [
        Line2D([0], [0], color=COLOR_X, label="$x_t$ (OGD)"),
        Line2D([0], [0], color=COLOR_XOPT, label="$x^\\star_t$"),
    ]
    fig.legend(handles=handles, loc='upper center', fontsize="large")
    fig.text(
        0.5, 0.8,
        "$f_t(x_t) = \\|x_t - x^\\star_t\\|_2^2$",
        ha="center", va="top", fontsize="large",
    )

    writer = matplotlib.animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=100*VIDEO_FPS)
    writer.setup(fig, "ogd.mp4")

    for i in range(T):
        render = (i % SKIP) == 0
        for plot in plots:
            plot.step(i, render)
        if render:
            writer.grab_frame()
            if VIDEO_FPS < 60:
                plt.show(block=False)
                plt.pause(1e-2)

    writer.finish()


if __name__ == "__main__":
    main()
