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
        self.axes.set_xlim(-20, 20)
        self.axes.set_ylim(-20, 20)

        plt.plot(self.ground[:, 0], self.ground[:, 1], color="black")

        ball1 = plt.Circle(
            self.trajectory[i, 0], radius=self.radiuss[0], color="yellow"
        )
        ball2 = plt.Circle(self.trajectory[i, 1], radius=self.radiuss[1], color="blue")
        ball3 = plt.Circle(self.trajectory[i, 2], radius=0, color="green")
        ball4 = plt.Circle(self.trajectory[i, 3], radius=0, color="red")

        self.axes.add_patch(ball1)
        self.axes.add_patch(ball2)
        self.axes.add_patch(ball3)
        self.axes.add_patch(ball4)

        plt.plot(
            [self.trajectory[i, 0, 0], self.trajectory[i, 1, 0]],
            [self.trajectory[i, 0, 1], self.trajectory[i, 1, 1]],
            color="black",
        )

        plt.plot(
            [self.trajectory[i, 0, 0], self.trajectory[i, 2, 0]],
            [self.trajectory[i, 0, 1], self.trajectory[i, 2, 1]],
            color="black",
        )

        plt.plot(
            [self.trajectory[i, 0, 0], self.trajectory[i, 3, 0]],
            [self.trajectory[i, 0, 1], self.trajectory[i, 3, 1]],
            color="black",
        )

        plt.plot(
            [self.trajectory[i, 1, 0], self.trajectory[i, 2, 0]],
            [self.trajectory[i, 1, 1], self.trajectory[i, 2, 1]],
            color="black",
        )

        plt.plot(
            [self.trajectory[i, 1, 0], self.trajectory[i, 3, 0]],
            [self.trajectory[i, 1, 1], self.trajectory[i, 3, 1]],
            color="black",
        )

        plt.plot(
            [self.trajectory[i, 2, 0], self.trajectory[i, 3, 0]],
            [self.trajectory[i, 2, 1], self.trajectory[i, 3, 1]],
            color="black",
        )

        return (
            ball1,
            ball2,
            ball3,
            ball4,
        )
