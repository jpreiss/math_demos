"""Branch-and-bound algorithm for binary linear programs.

Branch-and-bound is a general strategy for solving optimization problems that
would otherwise be convex, but some of the variables are constrained to lie in
a finite set of integers.

The algorithm traverses a search tree, where the root corresponds to the
original problem. "Branching" refers to the step where we grow the search
tree. We pick one integer variable and create a child node for each of its
possible values. Each child node therefore needs to solve an optimization
problem with one less integer variable.

"Bounding" refers to the key insight of the algorithm: at any node, we can
lower-bound (assuming a minimization problem, WLOG) the optimal value by
relaxing all of the free integer variables (the ones have not already been
fixed by branching) into real-valued variables. The solution will be at least
as good as the integer solution. If we keep track of the best integer solution
we have seen so far, and it is better than the relaxation value, then we can
immediately discard the entire subtree of the node.

This problem is NP-hard, but the branch/bound heuristic can still help prune
the search tree considerably. We demonstrate here on 0/1 knapsack problems. In
the output plot with a logarithmic y-axis, we see exponential growth of the
runtime for exhaustive search. The runtime of branch-and-bound also grows
exponentially but with a smaller exponent base. Although its constant factor is
over 100x larger for small n, it still begins outperforming exhaustive search
around n == 20.

This script takes about a minute to run.
"""

from dataclasses import dataclass, field, replace
from multiprocessing import Pool
import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


@dataclass
class BinaryLP:
    c: np.ndarray  # Maximize c^T x
    A: np.ndarray  # Subject to: Ax <= b.
    b: np.ndarray

    # Store y = accumulated value of c^T x for the elements of x that have
    # previously been fixed as integers.  We always process the variables in
    # order - if we used a more complex way to choose the branching variable,
    # we would need to store more information. Note: intelligently choosing the
    # branching variable is a big part of making b/b algorithms faster.
    y: float = 0.0
    x_ints: List[int] = field(default_factory=list)


# Results are tuple of (objective value, solution).
INFEASIBLE = (np.inf, [])

def solve(p: BinaryLP, best_so_far=np.inf):
    """Solve the binary LP. best_so_far is for recursion, do not supply."""

    # Recursive base case.
    if p.c.size == 0:
        if np.any(p.b < 0):
            # The last binary assignment made the program infeasible. See the
            # recursive call site for more information.
            return (np.inf, [])
        return (p.y, p.x_ints)

    # Solve the LP relaxation.
    bounds = np.tile([0.0, 1.0], (p.c.size, 1))
    try:
        res = linprog(p.c, p.A, p.b, bounds=bounds, options=dict(
            presolve=False,
        ))
    except ValueError as ex:
        # For some reason, scipy's linprog thinks certain causes of
        # infeasibility should raise an exception instead of returning an
        # "infeasible" status code. Who knows why...
        if "infeasible" in ex.args[0]:
            return INFEASIBLE
        raise

    # We should never see numerical issues for the examples in this script.
    if res.status == 1:
        assert False, "Iteration limit."
    if res.status == 4:
        assert False, "Numerical difficulties."

    # If the relaxation LP is infeasible, then the most recent binary branch
    # forced violation of a linear constraint.
    if res.status == 2:
        return INFEASIBLE

    # Unbounded.
    if res.status == 3:
        print("Unbounded!")
        return (-np.inf, [])

    # Add the extra objective cost from the fixed integer variables.
    relaxation_val = res.fun + p.y

    # If the solution to the relaxation is not better than the best integer
    # solution we have seen so far, then the integer solution cannot possibly
    # be better. Since our recursive ancestor already knows about the better
    # solution, no need to preserve any information.
    if relaxation_val > best_so_far:
        return INFEASIBLE

    # Construct the LP matrices for the problem with the first variable fixed.
    c = p.c[1:]
    A = p.A[:, 1:]
    y = p.y

    # Case x[0] = 0. No effect on y or b.
    p_x0 = BinaryLP(c, A, p.b, y, p.x_ints + [0])
    res_x0 = solve(p_x0, best_so_far)

    # Case x[0] = 1. Changes y and b.
    y_x1 = y + p.c[0]
    b_x1 = p.b - p.A[:, 0]
    p_x1 = BinaryLP(c, A, b_x1, y_x1, p.x_ints + [1])
    res_x1 = solve(p_x1, min(best_so_far, res_x0[0]))

    # Using tuple lexicographic sorting.
    return min(res_x0, res_x1)


def compare_times(n):
    npr = np.random.RandomState(n)

    # Sort weights to help our naive branching on variables in order.
    weights = np.flip(np.sort(npr.uniform(0.0, 1.0, size=n)))
    capacity = np.sqrt(np.sum(weights)) / 2.0

    # Run b/b algorithm.
    t0 = time.perf_counter()
    c = -weights
    A = weights.reshape((1, n))
    b = np.array([capacity])
    blp = BinaryLP(c, A, b)
    bb_result = solve(blp)
    bb_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Construct all 2^n x's for exhaustive search.
    xs = [[int(c) for c in f"{i:0{n}b}"] for i in range(2 ** n)]
    cs = np.array([c.T @ x for x in xs])
    # Discard infeasible solutions.
    cs[cs < -capacity] = 0.0
    best = np.argmin(cs)
    exh_time = time.perf_counter() - t0

    assert np.isclose(cs[best], bb_result[0], atol=1e-6)

    return bb_time, exh_time


def main():

    ns = np.arange(2, 23)
    pool = Pool(processes=os.cpu_count()-1)
    results = pool.map(compare_times, ns)
    bb_times, exh_times = zip(*results)

    plt.figure(figsize=(6, 4))
    plt.semilogy(ns, bb_times, label="branch / bound")
    plt.semilogy(ns, exh_times, label="exhaustive")
    plt.xticks(ns[::2])
    plt.grid(True)
    plt.legend()
    plt.xlabel("$n$")
    plt.ylabel("runtime (seconds)")
    plt.savefig("branchbound.pdf")


if __name__ == "__main__":
    main()
