"""Hedge is simple, elegant, and widely useful online learning algorithm.

Problem setting
---------------
We are given a set of N expert decision-makers who give us advice. We play a
game for T rounds. In round t, we select one expert choice_t and follow its
advice. Simultaneously (before observing our choice_t), the environment chooses
a loss value loss_t for each expert. We then observe the losses of all experts,
but "pay" the loss of only choice_t.

The goal is to minimize regret, defined as

  sum_{t = 1}^T loss_t(choice_t) - min_{k in 1...N} sum_{t = 1}^T loss_t(k).

In other words, we are competing against the best single expert in hindsight.
Note that we make no assumptions on loss_t. The environment is even allowed to
be adversarial: It knows our algorithm (but not our random seed!) and can pick
a worst-case sequence of loss_t to maximize our regret.

This problem setting is similar to multi-armed bandit (MAB) problems. The key
difference is that in MAB, we only observe the loss for the action (expert) we
took. In the expert-advice setting, we observe the loss of all experts. (The
adversarial MAB algorithm Exp3 is essentially plugging a statistical estimator
of the losses into Hedge.)

Algorithm
---------
The Hedge algorithm uses a simple exponential-weights rule -- a softmax
function applied to the cumulative sums of losses so far -- to define a
probability distribution over experts for each round. Our selection choice_t is
sampled from this distribution. For details, read the code.

Despite Hedge's remarkable simplicity, it can be shown that, with an
appropriately chosen learning rate parameter eta, Hedge's expected regret is
upper-bounded by sqrt(T log(N)). Therefore, the ratio regret/T goes to zero
as T goes to infinity.

In this demonstration, we apply Hedge to a non-stationary classification
problem. The true class is determined by a noisy linear decision boundary that
drifts a little bit each round. Our experts are a fixed set of randomly chosen
linear classifiers.

Looking at the output "hedge.pdf", we should observe the following:
 - Regret grows like sqrt(n) while mistakes grow approximately linearly.
 - Usually Hedge will converge on a single expert. If the policy at the last
   round has significant mass on more than one expert, those experts should
   both be in the top 5 and have very close losses.
 - Due to the drifting boundary, we will occasionally see Hedge begin to
   converge on one expert but then change to a different one. (Rerun a few
   times to get different random seeds and see this.)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


class Hedge:
    def __init__(self, n_experts, eta):
        """Initialize. Eta is a "learning rate" parameter - see below."""
        self.eta = eta
        self.loss_sums = np.zeros(n_experts)

    def tell_loss(self, loss):
        """Inform Hedge of the loss vector loss_t for this round."""
        self.loss_sums += loss

    def policy(self):
        """Get Hedge's probability distribution over actions."""
        exps = np.exp(-self.eta * self.loss_sums)
        return exps / np.sum(exps)

    def sample_action(self, npr):
        """Sample from Hedge's probability distribution over actions."""
        p = self.policy()
        return npr.choice(a=p.size, p=p)


def random_rotmat(npr, n, rotation_amount):
    """Sample a random n x n rotation matrix with approximately given angle."""
    # Sample from Lie algebra of SO(n) - skew symmetric matrices.
    tri = np.triu(npr.normal(size=(n, n)), k=1) / np.sqrt(n)
    r = tri - tri.T
    return expm(rotation_amount * r)


# This will only be used if you change dim to 2 and reduce T.
def plot_linear_decision_boundary(a, b, *args, **kwargs):
    """Plot 2 units' length of the line a^T x = b where a, x are 2D vectors."""
    orth = np.cross([a[0], a[1], 0], [0, 0, 1])[:2]
    R = np.column_stack([orth, a])
    x0 = R @ [-1, 0] + b * a
    x1 = R @ [1, 0] + b * a
    plt.plot([x0[0], x1[0]], [x0[1], x1[1]])


def main():
    n_experts = 60
    dim = 8
    rounds = 2000
    true_y_noise = 0.1
    # Unit-free parameter, related to how much the decision boundary normal
    # will rotate over the duration of the game.
    drift_amount = 0.5 / rounds
    eta = np.sqrt(np.log(n_experts) / rounds)

    npr = np.random.RandomState()

    # Sample the expert linear classifiers randomly.
    weights = npr.normal(size=(n_experts, dim))
    weights /= np.linalg.norm(weights, axis=1)[:, None]
    biases = npr.normal(size=(n_experts))

    # The true decision boundary will be linear, but corrupted by noise so it's
    # impossible for any expert to make zero mistakes, and drifting slowly over
    # time so we can't use methods that assume stationary loss distributions.
    drift_rotation = random_rotmat(npr, dim, drift_amount)
    drift_bias = 0.3 * drift_amount * npr.normal()

    true_weight = npr.normal(size=dim)
    true_weight /= np.linalg.norm(true_weight)
    true_bias = npr.normal()

    # Keep track of the policy distributions and loss vectors for later.
    policies = np.empty((rounds, n_experts))
    losses = np.empty((rounds, n_experts))
    our_losses = np.empty(rounds)

    # Play the game.
    hedge = Hedge(n_experts, eta)

    for i in range(rounds):
        policies[i] = hedge.policy()
        action = hedge.sample_action(npr)

        x = npr.normal(size=dim)
        true_y = true_weight.T @ x < true_bias + true_y_noise * npr.normal()
        expert_ys = weights @ x.T < biases
        loss = expert_ys != true_y

        our_losses[i] = loss[action]
        losses[i] = loss
        hedge.tell_loss(loss)

        true_weight = drift_rotation @ true_weight
        true_bias += drift_bias

        # Show the drifting decision boundary when low-dim and few rounds.
        if dim == 2 and rounds <= 100:
            plot_linear_decision_boundary(true_weight, true_bias)
            plt.axis("equal")
            plt.show(block=False)
            plt.pause(1e-3)


    # Apply hindsight to compute regret.
    hindsight_losses = np.sum(losses, axis=0)
    order_best = np.argsort(hindsight_losses)
    idx_best = order_best[0]
    ours = np.cumsum(our_losses)
    best = np.cumsum(losses[:, idx_best])
    regret = ours - best

    # Plot the cumulative mistakes and regret over the rounds.
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), tight_layout=True)
    axs[0].plot(ours, label="ours")
    axs[0].plot(best, label="best")
    axs[0].plot(regret, label="regret", color="black", linewidth=2)
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("mistakes")
    axs[0].legend()

    # Plot Hedge's expert distribution over time.
    axs[1].imshow(
        policies.T,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        origin="lower",
    )
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("policy")

    # Annotate distribution plot with the hindsight loss of the 5 best experts.
    message = "Τοp 5 experts\n"
    for i in order_best[:5]:
        message += f"{i:3}: loss = {hindsight_losses[i]:5.1f}\n"
    axs[1].text(rounds / 40, 0, message, fontfamily="monospace", fontsize="small")

    # Open the following PDF to see the results.
    fig.savefig("hedge.pdf")


if __name__ == "__main__":
    main()
