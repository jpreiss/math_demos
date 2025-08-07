from multiprocessing import Pool, cpu_count
import itertools as it

import matplotlib.pyplot as plt
import numpy as np


def random_spherical(rng, dim):
    x = rng.normal(size=dim)
    x *= np.sqrt(dim) / np.sqrt(np.sum(x ** 2))
    return x


def onepoint(rng, dim, radius, eta, query):
    x = np.zeros(dim)
    while True:
        yield x
        u = random_spherical(rng, dim)
        g = (u / radius) * query((x + radius * u)[None, :])
        x = x - 1.0 * eta * g


def twopoint(rng, dim, radius, eta, query):
    x = np.zeros(dim)
    while True:
        yield x
        u = random_spherical(rng, dim)
        q = query(np.stack([x - radius * u, x + radius * u]))
        diff = q[1] - q[0]
        g = (u / (2 * radius)) * diff
        x = x - 800 * eta * g


def residual(rng, dim, radius, eta, query):
    x = np.zeros(dim)
    prev = 0
    while True:
        yield x
        u = random_spherical(rng, dim)
        q = query((x + radius * u)[None, :])
        diff = q - prev
        prev = q
        g = (u / radius) * diff
        x = x - 30 * eta * g


class Env:
    def __init__(self):
        self.center = 0
        self.vel = 1
        self.targets = []
        self.costs = []

    def __call__(self, x):
        self.targets.append(self.center)
        assert len(x.shape) == 2
        y = np.sum((x - self.center) ** 2, axis=-1)
        self.costs.append(y[0])
        DT = 0.002
        self.vel -= DT * self.center
        self.center += DT * self.vel
        return y


def run(alg, seed):
    alg_fn = globals()[alg]
    T = 10000
    dim = 10
    rng = np.random.default_rng(seed)
    query = Env()
    gen = alg_fn(rng=rng, dim=dim, radius=0.10, eta=1e-4, query=query)
    traj = np.array(list(it.islice(gen, T)))
    cost = np.array(query.costs)
    target = np.array(query.targets)
    if dim > 1:
        traj = traj[:, 0]
    return traj, target, cost


def test_random():
    rng = np.random.default_rng()
    xs = np.stack([random_spherical(rng, 10) for _ in range(100000)])
    E = np.cov(xs.T)
    assert np.allclose(E, np.eye(10), atol=1e-2)


def main():
    fig, (ax_traj, ax_cost) = plt.subplots(1, 2, figsize=(10, 4))
    seeds = np.arange(30)
    pool = Pool(cpu_count() - 1)
    for alg in ["onepoint", "twopoint", "residual"]:
        print(alg)
        args = [(alg, seed) for seed in seeds]
        results = pool.starmap(run, args)
        label = alg
        color = None
        for traj, target, cost in results:
            handle, = ax_traj.plot(traj, label=label, color=color, alpha=0.1)
            color = handle.get_color()
            ax_cost.plot(np.cumsum(cost), label=label, color=color, alpha=0.1)
            label = None
    ax_traj.plot(target, linestyle="--", color=(0, 0.8, 0.2), linewidth=1, label="target")
    ax_traj.legend()
    ax_cost.legend()
    plt.savefig("bandit_convex.pdf")


if __name__ == "__main__":
    main()

