"""Illustrate how online gradient descent handles nonstationary environments.
"""

import sys

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def project(x):
    return np.clip(x, -1, 1)


def oracle(x, xopt):
    return 0.5 * np.sum((x - xopt)**2), x - xopt


def rot2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, -s],
        [s,  c]
    ])
    return R


def main(fast, slow, outpath):
    T = 200
    x = np.zeros(2)
    #xopt = np.zeros(2)

    xs = np.zeros((T, 2))
    xopts = np.zeros((T, 2))

    diameter = 2 * np.sqrt(2)
    lipschitz = 2 * np.sqrt(2)
    eta = diameter / (lipschitz * np.sqrt(T))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    opt_trace, = ax.plot([], [], label="opt", alpha=0.25, color="black")
    x_trace, = ax.plot([], [], label="alg", color="black")
    ax.legend()
    box = 1.5
    ax.set(xlim=[-box, box], ylim=[-box, box])
    ax.axis("equal")
    sns.despine(ax=ax, bottom=True, left=True)
    ax.set(xticks=[], yticks=[])

    writer = matplotlib.animation.FFMpegWriter(fps=24, bitrate=4000)
    writer.setup(fig, outpath)

    opt_slow = np.array([1, 0])
    opt_fast = np.array([0.25, 0])
    R_slow = rot2d(slow)
    R_fast = rot2d(fast)

    for i in range(T):
        x_trace.set_data(xs[:i, 0], xs[:i, 1])
        opt_trace.set_data(xopts[:i, 0], xopts[:i, 1])
        writer.grab_frame()
        plt.show(block=False)
        plt.pause(1e-2)
        xopt = opt_slow + opt_fast
        opt_slow = R_slow @ opt_slow
        opt_fast = R_fast @ opt_fast
        #xopt = xopt + 0.1 * np.random.normal(size=2)
        #xopt = project(xopt)
        xs[i] = x
        xopts[i] = xopt
        _, grad = oracle(x, xopt)
        x = project(x - eta * grad)

    writer.finish()

if __name__ == "__main__":
    fast = float(sys.argv[1])
    slow = float(sys.argv[2])
    outpath = sys.argv[3]
    main(fast, slow, outpath)
