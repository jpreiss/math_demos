import itertools as it
import numpy as np
import matplotlib.pyplot as plt


# roll out the linear dynamicx x' = Ax for T steps with time interval dt
def rollout(A, x, T, dt):
	out = np.zeros((T,) + x.shape)
	out[0] = x
	for t in range(1, T):
		out[t] = out[t-1] + dt * A @ out[t-1]
	return out


# construct a 2x2 matrix with the given trace and determinant
def trdet(tr, det):
	a = d = tr / 2
	bc = a*d - det
	b = c = np.sqrt(np.abs(bc))
	if bc < 0:
		b = -b
	return np.array([[a, b], [c, d]])


def main():

	tiles = 9
	points = 80
	steps = 200
	dt = 0.01
	traces = np.linspace(-2, 2, tiles)
	determinants = np.linspace(-2, 2, tiles)
	halfbox = (traces[1] - traces[0]) / 2.0

	plt.figure(figsize=(15, 15))

	for tr, det in it.product(traces, determinants):
		# create matrix, evaluate dynamics at random initial points
		A = trdet(tr, det)
		x0 = np.random.uniform(-halfbox, halfbox, size=(2, points))
		x = rollout(A, x0, steps, dt)

		# cut off points that escaped this tile's box
		for i in range(points):
			escaped = np.any(np.abs(x[:,:,i]) > halfbox, axis=1)
			x[escaped,:,i] = np.nan

		# plot many lines at once
		color = [0.2, (tr+2)/5, (det+2)/5]
		plt.plot(x[:,0,:] + tr, x[:,1,:] + det, color=color, linewidth=1)

	plt.box(False)
	plt.axis("equal")
	plt.gca().set_xticks([])
	plt.gca().set_yticks([])
	plt.savefig("phaseportraits.pdf")


if __name__ == "__main__":
	main()

