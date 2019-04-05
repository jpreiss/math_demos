import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

"""
Comparing accuracy of different ways to "integrate"
a sequence of axis-angle rotations into a sequence of rotation matrices.
Application: rotation estimate from a 3-axis gyroscope sensor.
Illustrates the Rodrigues exact formula for exponential map from so(3) to SO(3).
Compares against first- and second-order truncations of the matrix exponential.
"""

# the skew-symmetric matrix A s.t. Ay = x cross y
def skew(x):
	x, y, z = x
	return np.array([
		[0, -z, y],
		[z, 0, -x],
		[-y, x, 0]
	])

# first-order truncation of exponential map
def order1(axis, angle):
	K = skew(axis)
	return np.eye(3) + angle * K

# second-order truncation of exponential map
def order2(axis, angle):
	K = skew(axis)
	return np.eye(3) + angle * K + 0.5 * angle**2 * (K @ K)

# rodrigues closed form exact exponential map
# derived from properties of powers of K + the trig & exp power series
def rodrigues(axis, angle):
	K = skew(axis)
	return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

# "integrate" rotation matrix starting from the identity matrix
def integrate(axis, velocity, steps, integrator):
	mats = np.zeros((steps, 3, 3))
	R = np.eye(3)
	for i in range(steps):
		mats[i] = R
		R = R @ integrator(axis, velocity)
		U, E, Vt = la.svd(R)
		R = U @ Vt
	return mats

# sample a random point on the sphere
def random_axis():
	axis = np.random.normal(size=3)
	return axis / la.norm(axis)

def main():
	axis = random_axis()
	integrators = [rodrigues, order1, order2]
	velocities = [0.1, 0.2, 0.4]
	T = 100
	for vel in velocities:
		assert T * vel > np.pi

	for i, vel in enumerate(velocities):
		plt.subplot(len(velocities), 1, i + 1)
		for integrator in integrators:
			mats = integrate(axis, vel, T, integrator)
			plt.plot(mats[:,0,0], label=integrator.__name__)

		# trick to get scale only works if we go thru angle at least pi
		scale = (np.amax(mats[:,0,0]) - np.amin(mats[:,0,0])) / 2.0
		exact = scale * np.cos(vel * np.arange(T)) + (1.0 - scale)
		plt.plot(exact, label="true", color=(0, 0, 0), linestyle="dotted")
		plt.legend()

	plt.show()



if __name__ == "__main__":
    main()

