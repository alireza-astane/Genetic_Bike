import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Vis:
    def __init__(self, trajectory, radiuss, steps, ground):
        self.trajectory = trajectory
        self.radiuss = radiuss
        self.steps = steps
        self.ground = ground

    def run(self):
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        # afterwards, switch to zoomable GUI mode

        ani = animation.FuncAnimation(
            self.figure, self.animate, np.arange(1, self.steps), interval=1
        )
        plt.show()

    def animate(self, i):
        plt.cla()
        plt.gca().set_aspect('equal')
        plt.xlabel("x [m]")
        plt.ylabel("h [m]")

        self.axes.set_xlim(-2, 20)
        self.axes.set_ylim(-5, 20)

        plt.plot(
            [self.trajectory[i, 0, 0], self.trajectory[i, 1, 0]],
            [self.trajectory[i, 0, 1], self.trajectory[i, 1, 1]],
            color="tab:red",
        )

        plt.plot(
            [self.trajectory[i, 0, 0], self.trajectory[i, 2, 0]],
            [self.trajectory[i, 0, 1], self.trajectory[i, 2, 1]],
            color="tab:red",
        )

        plt.plot(
            [self.trajectory[i, 0, 0], self.trajectory[i, 3, 0]],
            [self.trajectory[i, 0, 1], self.trajectory[i, 3, 1]],
            color="tab:red",
        )

        plt.plot(
            [self.trajectory[i, 1, 0], self.trajectory[i, 2, 0]],
            [self.trajectory[i, 1, 1], self.trajectory[i, 2, 1]],
            color="tab:red",
        )

        plt.plot(
            [self.trajectory[i, 1, 0], self.trajectory[i, 3, 0]],
            [self.trajectory[i, 1, 1], self.trajectory[i, 3, 1]],
            color="tab:red",
        )

        plt.plot(
            [self.trajectory[i, 2, 0], self.trajectory[i, 3, 0]],
            [self.trajectory[i, 2, 1], self.trajectory[i, 3, 1]],
            color="tab:red",
        )

        plt.plot(self.ground[:, 0], self.ground[:, 1], 
            color="black", alpha=0.5, linewidth=10)

        ball1 = plt.Circle(self.trajectory[i, 0], radius=self.radiuss[0], 
            edgecolor=("tab:blue", 0.5), facecolor='none', linewidth=5)
        ball2 = plt.Circle(self.trajectory[i, 1], radius=self.radiuss[1], 
            edgecolor=("tab:blue", 0.5), facecolor='none', linewidth=5)
        ball3 = plt.Circle(self.trajectory[i, 2], radius=0.5, color="black")
        ball4 = plt.Circle(self.trajectory[i, 3], radius=0.5, color="black")

        self.axes.add_patch(ball1)
        self.axes.add_patch(ball2)
        self.axes.add_patch(ball3)
        self.axes.add_patch(ball4)

        return (
            ball1,
            ball2,
            ball3,
            ball4,
        )
