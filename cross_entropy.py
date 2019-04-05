import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_opt(np_random, eval_fn,
    init_pts, max_iters, max_std_stop, rank=0.1, smoothing=0.5):
    """
    Gradient-free semi-global minimization algorithm.
    see: "The cross-entropy method for optimization."
    Z. I. Botev et al., Handbook of Statistics, Elsevier, 2013.

    Args:
        np_random: an instance of numpy.random.RandomState.
        eval_fn: objective func to minimize, takes (N x dim), returns (N,)
        init_pts: initial guess of points. N is inferred from this
        max_iters: terminate after this many iterations regardless.
        max_std_stop: if max sqrt of covariance eigenvalues is <= this, stop.
        rank: proportion of points to keep to determine parameters of next iter.
        smoothing: new_smooth = smoothing * old + (1.0 - smoothing) * new

    Returns:
        mean, (cov sqrt eigvals, mean objective value, iteration)
    """

    assert len(init_pts.shape) == 2
    assert max_iters >= 1
    N, dim = init_pts.shape
    N_keep = int(rank * N)

    pts = init_pts
    mu = np.mean(pts, axis=0)
    cov = np.cov(pts.T)

    iter = 1
    while iter <= max_iters:
        # find the best points in this batch.
        obj_vals = eval_fn(pts)
        order = np.argsort(obj_vals)
        ibest = order[:N_keep]
        best = pts[ibest,:]
        best_meanval = np.mean(obj_vals[ibest])

        # find smoothed new gaussian parameters.
        mu = smoothing * mu + (1.0 - smoothing) * np.mean(best, axis=0)
        cov = smoothing * cov + (1.0 - smoothing) * np.cov(best.T)

        # check cov for convergence.
        stds = np.sqrt(np.linalg.eigvalsh(cov))
        if np.amax(stds) <= max_std_stop:
            break

        # sample new batch.
        pts = np_random.multivariate_normal(mu, cov, size=N)
        iter += 1

    return mu, (stds, best_meanval, iter)


#
# test on Rosenbrock function, well-known benchmark for optimization.
#
def cross_entropy_test_rosenbrock():

    def rosenbrock(x):
        xs, ys = x[:,0], x[:,1]
        a = 1.0
        b = 100.0
        return (a - xs)**2 + b*(ys - xs**2)**2

    # data for pseudo-contour plot (sqrt taken for better appearance)
    kgrid = 100
    t = np.linspace(-4, 4, kgrid)
    xgrid, ygrid = np.meshgrid(t, t)
    z = rosenbrock(np.column_stack([xgrid.flat, ygrid.flat])).reshape(kgrid, kgrid)

    # scatter plot each iteration w/ contour plot overload
    def vis(x):
        plt.clf()
        plt.contour(xgrid, ygrid, np.sqrt(z))
        plt.plot(x[:,0], x[:,1], "ko")
        plt.axis("equal")
        plt.xlim([t[0], t[-1]])
        plt.ylim([t[0], t[-1]])
        plt.show(block=False)
        plt.pause(0.1)

    def eval_fn(x):
        vis(x)
        return rosenbrock(x)

    npts = 512
    rank = 0.1
    npr = np.random.RandomState()
    init_mu = np.array([-2, -1])
    init_cov = 4**2 * np.eye(2)
    init_pts = npr.multivariate_normal(init_mu, init_cov, size=npts)

    mu_opt, (stds_opt, optval, iters) = cross_entropy_opt(npr, eval_fn,
        init_pts, max_iters=1000, max_std_stop=0.01, rank=rank)

    # global optimum of rosenbrock function is at (1, 1)
    np.set_printoptions(precision=4)
    print("cross-entropy method")
    print(f"found approx optimum of rosenbrock func in {iters} iterations")
    print(f"mu = {mu_opt}, stds = {stds_opt}, value = {optval:.7f}")
    print("true optimum value is 0 at [1, 1].")
    print("(close pyplot window to exit.)")

    plt.show(block=True)


if __name__ == "__main__":
    cross_entropy_test_rosenbrock()

