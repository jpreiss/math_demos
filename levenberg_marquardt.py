from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

"""
Demonstrate the Levenberg-Marquardt algorithm
for nonlinear least-squares curve fitting problems.
Levenberg-Marquardt is an "introspective" algorithm that
balances second-order and first-order steps
depending on their success at reducing the residual.
"""

# x, y : 1d curve dataset
# params : initial guess for curve parameters
# f: function f(x, params) -> y
# jacobian: j(x, params) - > deriv of sum squared error of f wrt params
#
def levenberg_marquardt(x, y, params, f, jacobian):
	L = 0.01
	L_step = 2
	MAX_ITERS = 500
	EPSILON = 1e-15 # termination condition - see below
	residual = y - f(x, params)
	err = np.sum(residual ** 2)
	for i in range(MAX_ITERS):
		j = jacobian(x, params)
		jTj = j.T.dot(j)
		delta = np.linalg.solve(jTj + L * np.diag(jTj), j.T.dot(residual))
		if np.mean(np.abs(delta)) < EPSILON:
			return params, i, "gradient < {}".format(EPSILON)
		test_params = params + delta
		test_residual = y - f(x, test_params)
		test_err = np.sum(test_residual ** 2)
		if np.abs(test_err - err) < EPSILON:
			return params, i, "residual change < {}".format(EPSILON)
		if test_err < err:
			params, residual, err = test_params, test_residual, test_err
			# second-order step was good, use more second-order information
			L = L / L_step
		else:
			L = L * L_step
	return params, i, "{} iters without convergence".format(MAX_ITERS)


# function and derivative - y = ax^p with a, p unknown reals
def f(x, b):
	return b[0] * x ** b[1]

def jacobian(x, b):
	j0 = x ** b[1]
	j1 = b[0] * (x ** b[1]) * np.log(x)
	nr = x.size
	out = np.hstack([j0.reshape(nr,1), j1.reshape(nr,1)])
	assert(out.shape[1] == 2)
	return out

def main():
	# generate data to be fitted
	#b_true = 5 * np.random.rand(1) + 6
	b_true = 3 * np.random.rand(2) + 3
	N = 40
	x = 0.1 + np.random.rand(N)
	y = f(x, b_true)
	# uncomment to add noise - param fit will no longer match true
	#y += 0.1 * np.std(y) * np.random.normal(size=N)

	# run nonlinear least squares algorithm
	b_guess = np.array([3, 3])
	soln, iters, status = levenberg_marquardt(x, y, b_guess, f, jacobian)

	print("levenberg-marquardt: {} iters, status = {},".format(iters, status))
	print("        result:", soln)
	print("true parameter:", b_true)

	plt.scatter(x, y)
	xlin = np.linspace(0, 1.2, 100)
	plt.plot(xlin, f(xlin, soln))
	plt.show()


if __name__ == '__main__':
	main()

