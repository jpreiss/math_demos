"""Animation of the Frank-Wolfe algorithm for convex optimization.

The Frank-Wolfe algorithm minimizes a differentiable convex function f over a
compact convex set K. Iterations take the form

    y = min_{v ∈ K} ⟨v, ∇f(x)⟩
    x' = x + η(y - x),

where η is a scalar rate. The solution to the linear subproblem is always on
the boundary of K. Therefore, the k'th iterate x_k is always a convex
combination of the initial guess and no more than k boundary points of K. This
property can be very advantageous for applications such as sparse
reconstruction or convex relaxations of combinatorial problems. In problems
where K is highly structured, the linear subproblem can often be solved more
quickly than a generic linear programming solver.

In this code, we use Frank-Wolfe to solve a quadratic program over a random
polytope.  We animate each step of the algorithm. Frank-Wolfe does not perform
particularly well in this kind of problem because it is prone to taking large
steps when near the optimum.
"""

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial


def plotseg(x0, x1, plot=None, **kwargs):
    """Plots a line segment from x0 to x1. Forwards kwargs for plot()."""
    x = (x0[0], x1[0])
    y = (x0[1], x1[1])
    if plot is None:
        return plt.plot(x, y, **kwargs)
    else:
        plot.set_data(x, y)


def func_contour(f, xlim, ylim, n_contours=20):
    """Plots contours of an arbitrary function."""
    tx = np.arange(*xlim, 0.01)
    ty = np.arange(*ylim, 0.01)
    x, y = np.meshgrid(tx, ty)
    xy = np.column_stack([x.flat, y.flat])
    z = f(xy).reshape(x.shape)
    levels = np.linspace(1e-1, np.amax(z.flat), n_contours)
    return plt.contour(x, y, z, levels=levels, colors="black", linewidths=0.5)


def main():
    # Generate random vertices in sorted order.
    n_points = 10
    points = np.random.normal(size=(n_points, 2))
    hull = sp.spatial.ConvexHull(points)

    # ConvexHull stores in sorted order, suitable for polygon.
    vertices = hull.points[hull.vertices]

    # Generate a random quadratic problem min_x |Ax - b|_2^2.
    b = np.random.normal(size=2)
    A = np.random.normal(size=(2, 2)) + np.eye(2)

    def f(x):
        if len(x.shape) < 2:
            x = x[None, :]
        return 0.5 * np.sum((x @ A - b[None, :]) ** 2, axis=1).squeeze()

    gf = jax.grad(f)

    # Make the window not steal focus whenever it's updated.
    plt.rcParams["figure.raise_window"] = False
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    pause_time = 0.01

    color_fill = 0.8 * np.ones(3)
    color_grad = [0.1, 0.2, 1.0]
    color_step = [0.1, 0.8, 0.4]

    # Plot the polygon.
    patch = mpl.patches.Polygon(vertices, facecolor=color_fill, edgecolor="black")
    ax.add_patch(patch)
    ax.plot(
        vertices[:, 0],
        vertices[:, 1],
        marker="o",
        linewidth=0,
        color="black",
        markersize=3,
    )
    ax.axis("equal")

    # Plot the unconstrained optimum..
    argmin = np.linalg.solve(A.T, b)
    (plot_opt_marker,) = plt.plot(
        *argmin, marker="x", color="red", markersize=10, linewidth=0
    )

    # Contour plot the objective.
    xlim = plt.xlim()
    ylim = plt.ylim()
    func_contour(f, xlim, ylim)

    # Set up the plot elements before looping.
    x = np.zeros(2)
    (plot_x_marker,) = plt.plot(*x, marker="o", color="black", label="$x$", linewidth=0)

    g = gf(x)
    (plot_gradient,) = plotseg(
        x, x - 0.25 * g, linewidth=2, color=color_grad, label="$\\nabla f(x)$"
    )

    vert_min = vertices[0]
    (plot_vert_seg,) = plotseg(
        x, vert_min, linestyle="--", linewidth=0.5, color="black"
    )
    (plot_vert_marker,) = plt.plot(
        *vert_min, marker="o", color="cyan", markersize=5, label="$v$", linewidth=0
    )

    (plot_step_seg,) = plotseg(x, x, linewidth=2, color=color_step, label="step")

    plt.legend()
    plt.show(block=False)

    iters = 100
    for it in range(iters):

        plot_x_marker.set_data(x)
        plt.pause(pause_time)

        # Plot the gradient.
        g = gf(x)
        plotseg(x, x - 0.25 * g, plot=plot_gradient)
        plt.pause(pause_time)

        # The core of the algorithm. Note we solve the LP by an exhaustive
        # search over vertices, instead of the typical LP solver that works
        # with a polytope in halfspace form. This is meant as an analogy to the
        # "highly structured" case discussed in the file docstring.
        inners = vertices @ g.T
        idx_min = np.argmin(inners)
        vert_min = vertices[idx_min]
        plotseg(x, vert_min, plot=plot_vert_seg)
        plot_vert_marker.set_data(vert_min)
        plt.pause(pause_time)

        # Update our guess.
        eta = 2.0 / (it + 2)
        step = eta * (vert_min - x)
        plotseg(x, x + step, plot=plot_step_seg)
        plt.pause(pause_time)
        x += step


if __name__ == "__main__":
    main()
