"""Compare methods for large linear least-squares regression problems.

Exact methods based on linear algebra become slow for high-dimensional
inputs/outputs and/or huge amounts of data. Stochastic gradient descent (SGD)
iteratively finds an approximate solution by performing gradient descent on the
least-squares objective with a small subset of the full data at each iteration.
SGD is guaranteed to approximately converge due to the strong convexity of the
linear least-squares problem. SGD consumes less memory and runs faster for
large problems if the rate is tuned correctly.
"""

import multiprocessing.pool
import time

import numpy as np
import matplotlib.pyplot as plt


def error(X, Y, W):
    """Mean L2 error for multi-output linear least-squares."""
    errs = np.sum((Y - X @ W) ** 2, axis=1)
    return np.mean(errs)


def lstsq_exact(X, Y):
    """Linear least-squares using exact (SVD-based) method."""
    t0 = time.time()
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    t = time.time() - t0
    return t, W


def lstsq_sgd(X, Y, rate, batch, iters):
    """Linear least-squares using stochastic gradient descent."""
    samples, dim_in = X.shape
    _, dim_out = Y.shape
    W = np.zeros((dim_in, dim_out))
    Ws = [W]
    t0 = time.time()
    for _ in range(iters):
        idx = np.random.choice(samples, size=batch, replace=False)
        g = -X[idx].T @ (Y[idx] - X[idx] @ W) / batch
        W = W - rate * g
        Ws.append(W)
    t = time.time() - t0
    return t, Ws


def main():
    samples = 100000
    dim_in = 1000
    dim_out = 10

    # SGD hyperparameters.
    rate = 1e1
    batch = 200
    iters = 2000

    # Generate some random data as linear + noise.
    X = np.random.normal(size=(samples, dim_in)) / np.sqrt(dim_in)
    W_true = np.random.normal(size=(dim_in, dim_out))
    Y = (X @ W_true) + 0.1 * np.random.normal(size=(samples, dim_out))

    # Compute the true least-squares solution in a separate process.
    pool = multiprocessing.pool.Pool()
    exact_result = pool.starmap_async(lstsq_exact, [(X, Y)])

    # Compute the SGD solution.
    time_sgd, Ws = lstsq_sgd(X, Y, rate, batch, iters)

    # Subsample the error plot because computing exact errors is expensive.
    t = (time_sgd / iters) * np.arange(iters + 1)
    subsample = np.arange(0, iters + 1, 20)
    t = t[subsample]
    errors = [error(X, Y, Ws[i]) for i in subsample]

    # Wait for the least-squares process to finish.
    time_lstsq, W_lstsq = exact_result.get()[0]
    err_lstsq = error(X, Y, W_lstsq)

    # Construct the plot.
    plt.rcParams["text.usetex"] = True
    plt.figure(figsize=(4.5, 3.0), tight_layout=True)
    plt.semilogy(t, errors, color="red", label="SGD")
    plt.axvline(time_lstsq, color="blue", linewidth=0.5, label="Batch time")
    plt.axhline(err_lstsq, color="black", linestyle=":", label="Batch accuracy")
    plt.xlabel("Wall-clock time")
    plt.ylabel("Mean $\\|x - \\hat x\\|_2^2$")
    plt.legend()
    plt.savefig("linear_regression_methods.pdf")


if __name__ == "__main__":
    main()
