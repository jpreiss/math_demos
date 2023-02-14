"""Illustrate how online gradient descent handles nonstationary environments.
"""

def project(x):
    return np.clip(x, -1, 1)


def oracle(x, xopt):
    return 0.5 * np.sum((x - opt)**2), x - xopt


def main():
    T = 100
    x = np.zeros(2)
    xopt = np.zeros(2)

    xs = np.zeros((T, 2))
    xopts = np.zeros((T, 2))

    eta = 1.0 / np.sqrt(T)

    fig, ax = plt.subplots(1, 1)


    for i in range(T):
        xopt = project(xopt + 0.1 * np.random.normal(size=2))
        _, grad = oracle(x, xopt)

