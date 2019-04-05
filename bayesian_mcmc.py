import numpy as np
import numpy.linalg as LA
import itertools as it
import matplotlib.pyplot as plt

"""
Demonstration of Markov Chain Monte Carlo Bayesian inference.
Bayesian MCMC is a technique for approximating the
posterior distribution over parameters of some data-generating process
given a data set and, optionally, a prior distribution over parameters.
This toy problem demonstrates with a 2D Gaussian distribution.
However, Bayesian MCMC is most useful when the parameter space
is high-dimensional and the above prior has no analytic solution.
"""

# Metropolis algorithm, a simple MCMC sampling procedure.
# implemented as generator function, use itertools.islice
def metropolis(log_pdf, sampler, x0):
	x = x0
	while True:
		xp = sampler(x)
		log_ratio = log_pdf(xp) - log_pdf(x)
		u = np.random.uniform()
		if np.log(u) < log_ratio:
			x = xp
		yield x


# create a function that computes the log probability density
# of each point in a data set (points are rows)
def gaussian_log_pdf(mu, sigma):
	sinv = LA.inv(sigma)
	mu = np.array(mu)
	def logpdf(x):
		d = x - mu
		prod = np.sum(np.dot(d, sinv) * d, axis=1)
		scl = LA.det(2 * np.pi * sigma) ** -0.5
		return np.log(scl) - 0.5 * prod
	return logpdf


# main
def metro_bayes():
	# the true parameters of the data
	mu = np.array([1,2])
	sigma = np.diag([0.05, 1])

	# generate the data
	kd = 200
	data = np.dot(np.random.normal(size=(kd,2)), LA.cholesky(sigma)) + mu

	# function to compute probability of (mu, sigma) params
	# given the data. We could also add a prior here if desired.
	def log_pdf(params):
		mu = params[:2]
		sigma = np.diag(params[2:])
		if np.any(params[2:] <= 0.0001):
			return -np.inf
		g = gaussian_log_pdf(mu, sigma)
		logp = np.sum(g(data))
		return logp

	# function that takes a step in the parameter space for Metropolis.
	# step size needs to be tuned to number of data points (kd above).
	# with more data, pdf of parameters is sharper, so step needs to be smaller,
	# otherwise Metropolis will never take a step once it reaches
	# a high-probability region.
	def sampler(x):
		return x + 0.01*np.random.normal(size=4)

	# initial guess unit normal
	x0 = np.array([0, 0, 1, 1])
	k = 10000
	m = np.row_stack(it.islice(
		metropolis(log_pdf, sampler, x0),
		k))
	# remove first chunk of data for MCMC burn-in
	m = m[2000:,:]

	# histogram the posterior distributions of the parameters.
	names = ("${\mu_0}$", "${\mu_1}$", "${\sigma_0}$", "${\sigma_1}$")
	true_vals = np.concatenate([mu, np.diag(sigma)])
	for i in range(4):
		plt.subplot(2, 2, i + 1)
		plt.hist(m[:,i])
		plt.axvline(x=true_vals[i], color=(1,0,0))
		# no idea why pyplot reverses these
		plt.legend(["true value", "MCMC posterior"])
		plt.xlabel(names[i], FontSize=16)
		plt.ylabel("count")
	plt.show()

metro_bayes()
