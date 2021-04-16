import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation

"""
LQR (Linear-Quadratic Regulator) optimal control
of the linearized cart-pole balancing problem.
we follow the notation and coordinate systems of 
"The pole balancing problem: a benchmark control theory problem"
by Jason Brownlee, technical report, 2005.
"""

# physical constants:
mc = 1.0 # mass cart
mp = 0.1 # mass pole
l = 0.5  # length of pole
g = 9.81 # gravity

# system state global, because python's closures suck
th = 0.50
dth = 0
x = -1
dx = 0


def dynamics(th, dth, x, dx, force, dt):
    """Integrate nonlinear dynamics of cart-pole system."""
    sth = np.sin(th)
    cth = np.cos(th)

    # ODEs
    ddth = ((g*sth + cth*(-force - mp*l*dth**2*sth)/(mc + mp))
         / (l*(4/3 - (mp*cth**2)/(mc + mp))))

    ddx = (force + mp*l*(dth**2*sth - ddth*cth))/(mc + mp)

    # noise
    ddth += 0.4 * np.random.normal()
    ddx += 0.4 * np.random.normal()

    # euler integration
    dth = dth + dt * ddth
    th = th + dt * dth
    dx = dx + dt * ddx
    x = x + dt * dx

    return th, dth, x, dx


def lqr(A, B, Q, R):
    """Compute LQR optimal feedback control gains.

    Args:
        A, B: Continuous system dynamics dx/dt = Ax + Bu.
        Q, R: Quadratic cost J(t) = x'Q x + u'R u.

    Returns:
        K: Gain matrix.
    """
    B = B.reshape((B.size, 1))
    P = sp.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, np.dot(B.T, P))
    return K


def cartpole_linear():
    """Construct linearized dynamics matrices for the cart-pole system.

    Derivation:

    ddth = (g*0 + 1*(-force - 0)/(mc + mp) / (l*(4/3 - mp/(mc+mp)))
         = C * force, where C = -1/(mc+mp) / (l*4/3 - mp/(mc+mp))
    ddx = (force + mp*l*(0-ddth)/(mc + mp)
        = force - mp*l*C*force/(mc+mp)
        = (1 - mp*l*C/(mc+mp)) * force = C2 * force

    s = [th dth x dx]
    ds/dt = As + Bforce, where:
    A = [0    1 0 0
         grav 0 0 0
         0    0 0 1
         0    0 0 0],

    B = [0 C 0 C2].

    Returns:
        A, B: Dynamics matrices such that $\\dot x \\approx Ax + Bu$.
    """

    A = np.array([
        [0,    1, 0, 0],
        [9.81, 0, 0, 0],
        [0,    0, 0, 1],
        [0,    0, 0, 0]
    ])
    C1 = (-1/(mc+mp)) / (l*4/3 - mp/(mc+mp))
    C2 = 1 - mp*l*C1/(mc+mp)
    B = np.array([0, C1, 0, C2])
    return A, B


def simulate(K):
    # for rendering
    fps = 30
    dt = 1.0 / fps
    cart_w = 0.3
    cart_h = 0.15
    track_limit = 2.4
    xlimits = [-track_limit, track_limit]

    secs = 5
    frames = fps * secs

    fig, ax = plt.subplots()
    line, = ax.plot([0, 0], [1, 1])
    rect = mpl.patches.Rectangle((0, 0), cart_w, cart_h)
    ax.add_patch(rect)
    ax.axis("equal")
    ax.set_xlim(xlimits)
    ax.set_ylim([-.2*l, 2.2*l])

    def anim_fn(iframe):
        global x, dx, th, dth
        line.set_xdata([x, x + 2*l*np.sin(th)])
        line.set_ydata([0, 2*l*np.cos(th)])
        rect.set_xy((x-cart_w/2, -cart_h/2))
        force = np.dot(K, np.array([th, dth, x, dx]))[0]
        th, dth, x, dx = dynamics(th, dth, x, dx, force, dt)
        return line, rect

    ani = mpl.animation.FuncAnimation(fig, anim_fn,
        interval=1000*dt, frames=frames, repeat=False,
    )
    plt.show()


def main():
    A, B = cartpole_linear()
    Q = np.diag([1, 0.1, 1, 0.1])
    R = 0.001 * np.eye(1)
    K = lqr(A, B, Q, R)
    simulate(K)


if __name__ == "__main__":
    main()
