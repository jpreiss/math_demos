import itertools as it

import matplotlib.pyplot as plt
import numpy as np


def onepoint(rng, dim, radius, eta, query):
    x = np.zeros(dim)
    while True:
        yield x
        u = rng.normal(size=dim)
        g = (1.0 / radius) * query(x + radius * u) * u
        x = x - 0.1 * eta * g


def twopoint(rng, dim, radius, eta, query):
    x = np.zeros(dim)
    while True:
        yield x
        u = rng.normal(size=dim)
        q = query(np.stack([x - radius * u, x + radius * u]))
        diff = q[1] - q[0]
        g = (u / (2 * radius)) * diff
        x = x - 10 * eta * g


def residual(rng, dim, radius, eta, query):
    x = np.zeros(dim)
    prev = 0
    while True:
        yield x
        u = rng.normal(size=dim)
        q = query(x + radius * u)
        diff = q - prev
        prev = q
        g = (u / radius) * diff
        x = x - 2 * eta * g


class Env:
    def __init__(self):
        self.center = 1
        self.vel = 0
        self.targets = []
        self.costs = []

    def __call__(self, x):
        self.targets.append(self.center)
        y = (x - self.center) ** 2
        self.costs.append(np.atleast_1d(y)[0])
        DT = 0.005
        self.vel -= DT * self.center
        self.center += DT * self.vel
        return y


def run(alg, seed):
    T = 10000
    rng = np.random.default_rng(seed)
    query = Env()
    gen = alg(rng=rng, dim=1, radius=0.01, eta=3e-3, query=query)
    traj = np.array(list(it.islice(gen, T)))
    cost = np.array(query.costs)
    target = np.array(query.targets)
    return traj, target, cost


def main():
    seed = 0
    fig, (ax_traj, ax_cost) = plt.subplots(2, 1, figsize=(6, 6))
    for alg in [onepoint, twopoint, residual]:
        traj, target, cost = run(alg, seed)
        ax_traj.plot(traj, label=alg.__name__)
        ax_cost.plot(np.cumsum(cost), label=alg.__name__)
    ax_traj.plot(target, linestyle="--", color=(0, 0.8, 0.2), linewidth=1, label="target")
    plt.legend()
    plt.savefig("bandit_convex.pdf")


if __name__ == "__main__":
    main()

