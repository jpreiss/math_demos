import numpy as np
import matplotlib.pyplot as plt

"""Demonstrate total variation (TV) denoising of a 1D signal.

TV denoising assumes that a signal is mostly piecewise constant. It solves the
following unconstrained optimization problem:

    minimize L(y) = sum (x[i] - y[i])^2 + alpha * sum abs(y[i+1] - y[i])

where x is the original signal, y is the decision variable, and alpha is the
"strength" parameter. Here, we solve the problem via gradient descent with
hand-computed gradient.
"""

def tv_denoise(x, alpha):
    x = np.concatenate([[x[0]], x])
    y = 0 + x
    descent_rate = 0.0005 / alpha
    iters = 3000

    # Compute the gradient of the TV term of the objective function L. Note
    # that we call this function twice - each variable contributes to both its
    # forward and backward difference terms in L.
    def tvgrad(diff):
        # Gradient of absolute value causes oscillation near zero. Replace it
        # with this soft absolute value instead.
        eps = 1e-6
        g = diff / np.sqrt(diff ** 2 + eps)
        return g

    objvals = []
    i = 0
    for i in range(iters):
        fwd_diff = y[1:-1] - y[:-2]
        bwd_diff = y[1:-1] - y[2:]
        grad_tv = alpha * (tvgrad(fwd_diff) + tvgrad(bwd_diff))
        grad_err = (y - x)[1:-1]
        objval = np.sum(alpha * np.abs(fwd_diff) + 0.5 * grad_err ** 2)
        objvals.append(objval)
        y[1:-1] -= descent_rate * (grad_tv + grad_err)
        # Ensure that boundary values don't contribute to loss.
        y[0] = y[1]
        y[-1] = y[-2]
        i += 1
    x = x[1:-1]
    y = y[1:-1]

    return y, objvals


def main():
    N = 1000
    t = np.arange(1000)
    signal = 1.0 * (t < 500) + 0.1 * np.random.normal(size=N)
    signal -= np.mean(signal)

    alphas = [0.1, 0.3, 1.0, 10.0]
    colors = plt.cm.cool(np.linspace(0, 1, len(alphas)))

    fig, (ax_loss, ax_sig) = plt.subplots(2, 1, figsize=(8, 10))
    ax_sig.plot(signal, '0.7', color=(0.8, 0.8, 0.8), label="signal")

    for alpha, color in zip(alphas, colors):
        denoised, objvals = tv_denoise(signal, alpha)
        label = f"$\\alpha = {alpha:.1f}$"
        ax_loss.plot(np.log(objvals), color=color, label=label)
        ax_sig.plot(denoised, color=color, label=label)

    ax_loss.set_ylabel("log(loss)")
    ax_loss.set_xlabel("iteration")
    ax_loss.legend()

    ax_sig.set_xlabel("time")
    ax_sig.set_ylabel("signal")
    ax_sig.legend()

    print("hello")
    fig.savefig("total_variation_denoising.pdf")


main()
