from copy import deepcopy

import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np


class PlanarArmIK:
    def __init__(self):
        self.th1 = 0.0
        self.th2 = 0.3
        self.r1 = 1.0
        self.r2 = 0.8

    def position_and_jacobian(self):
        th1, th2, r1, r2 = self.th1, self.th2, self.r2, self.r2
        c1 = np.cos(th1)
        s1 = np.sin(th1)
        c12 = np.cos(th1 + th2)
        s12 = np.sin(th1 + th2)
        J = np.array([
            [-r1*s1 - r2*s12, -r2*s12],
            [ r1*c1 + r2*c12,  r2*c12],
        ])
        x = [0.0, r1*c1, r1*c1 + r2*c12]
        y = [0.0, r1*s1, r1*s1 + r2*s12]
        return x, y, J

    def ik_step(self, goal, gain, dt):
        x, y, J = self.position_and_jacobian()
        pos = [x[-1], y[-1]]
        to_goal = goal - pos

        lam = 1e-1
        reg_matx = lam * np.diag([2.0, 1.0])
        ridge_pinv = np.linalg.inv(J.T @ J + reg_matx.T @ reg_matx.T) @ J.T

        #dth1, dth2 = np.linalg.pinv(J) @ to_goal
        dth1, dth2 = ridge_pinv @ to_goal

        self.th1 += gain * dt * dth1
        self.th2 += gain * dt * dth2

        return x, y

    def ik_multi(self, goal, gain, dt, iters):
        ik = deepcopy(self)
        for _ in range(iters):
            ik.ik_step(goal, 1.0, dt)
        self.th1 += gain * dt * (ik.th1 - self.th1)
        self.th2 += gain * dt * (ik.th2 - self.th2)
        x, y, _ = self.position_and_jacobian()
        return x, y


class PlanarArmPlot:
    def __init__(self):
        self.ik = PlanarArmIK()
        self.dt = 0.1
        self.gain = 5.0
        self.fig = None
        self.line = None
        self.circle = None
        self.goal = np.array([1, 0])

    def interactive(self):
        self.fig, ax = plt.subplots(1, 1)
        ax.axis("equal")
        self.circle = mpl.patches.Circle((0, 0), 0.1)
        ax.add_artist(self.circle)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        x, y, _ = self.ik.position_and_jacobian()
        (self.line,) = ax.plot(x, y, linewidth=6)

        anim = mpl.animation.FuncAnimation(self.fig, self.tick,
            interval=1000*self.dt, frames=1000, repeat=False,
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.motion)
        plt.show()

    def motion(self, event):
        # TODO handle properly
        goal = np.array([event.xdata, event.ydata])
        if None in goal:
            return
        self.goal = goal

    def tick(self, event):
        #x, y = self.ik.ik_step(self.goal, self.gain, self.dt)
        x, y = self.ik.ik_multi(self.goal, self.gain, self.dt, iters=100)
        pos = [x[-1], y[-1]]
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.circle.center = pos
        self.fig.canvas.draw()


def main():
    arm = PlanarArmPlot()
    arm.interactive()


if __name__ == "__main__":
    main()
