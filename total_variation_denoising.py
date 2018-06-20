import numpy as np
import matplotlib.pyplot as plt

"""
Total variation denoising assumes that a signal is mostly piecewise constant.
It solves the following unconstrained optimization problem:
    minimize L(y) = sum (x[i] - y[i])^2 + alpha * sum abs(y[i+1] - y[i])
where x is the original signal, y is the decision variable,
and alpha is the "strength" parameter.
Here, we solve the problem via gradient descent with hand-computed gradient.
"""

def tv_denoise(x, alpha):
	x = np.concatenate([[x[0]], x])
	y = 0 + x
	step = 0.0005 / alpha

	# compute the gradient of the TV term of the objective function L.
	# note that we call this function twice - each variable contributes
	# to both its forward and backward difference terms in L.
	def tvgrad(diff):
		# gradient of absolute value causes oscillation near zero
		# replace with this soft absolute value instead
		eps = 1e-6
		g = diff / np.sqrt(diff ** 2 + eps)
		return g

	objvals = []
	i = 0
	#while len(objvals) < 2 or abs(objvals[-2] - objvals[-1]) / objvals[-2] > 1e-4:
	for i in range(3000):
		fwd_diff = y[1:-1] - y[:-2]
		bwd_diff = y[1:-1] - y[2:]
		grad_tv = alpha * (tvgrad(fwd_diff) + tvgrad(bwd_diff))
		grad_err = (y - x)[1:-1]
		objval = np.sum(alpha * np.abs(fwd_diff) + 0.5 * grad_err ** 2)
		objvals.append(objval)
		y[1:-1] -= step * (grad_tv + grad_err)
		# perturbations to neutralize influence of boundary values
		y[0] = y[1] + 0.0000000001 * ((i % 2) - 0.5)
		y[-1] = y[-2] + 0.0000000001 * ((i % 2) - 0.5)
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

	plt.subplot(2,1,2)
	plt.hold(True)
	plt.plot(signal, '0.7', color=(0.8, 0.8, 0.8))

	for alpha, color in zip(alphas, colors):
		denoised, objvals = tv_denoise(signal, alpha)

		plt.subplot(2,1,1)
		plt.hold(True)
		plt.plot(np.log(objvals), color=color)
		plt.ylabel("log(obj value)")
		plt.xlabel("iteration")

		plt.subplot(2,1,2)
		plt.plot(denoised, color=color)
		plt.xlabel("time")
		plt.ylabel("signal")

	labels = ["alpha = {:.1f}".format(a) for a in alphas]
	plt.subplot(2,1,1)
	plt.legend(labels)
	plt.subplot(2,1,2)
	plt.legend(["signal"] + labels)
	plt.show()


main()
