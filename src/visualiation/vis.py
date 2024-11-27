import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define the figure hierarchy (figure holds axes)
figure = plt.figure()
axes = figure.add_subplot(111)


# required data for the animation
time_steps_example = 100
trajectory_example = np.zeros((time_steps_example, 4, 2))

trajectory_example[:, 0] = np.array(
    [
        np.sin(np.linspace(0, 2 * np.pi, time_steps_example) + 1),
        np.cos(np.linspace(0, 2 * np.pi, time_steps_example) + 1),
    ]
).T
trajectory_example[:, 1] = np.array(
    [
        np.sin(np.linspace(0, 2 * np.pi, time_steps_example) + 2),
        np.cos(np.linspace(0, 2 * np.pi, time_steps_example) + 2),
    ]
).T
trajectory_example[:, 2] = np.array(
    [
        np.sin(np.linspace(0, 2 * np.pi, time_steps_example) + 3),
        np.cos(np.linspace(0, 2 * np.pi, time_steps_example) + 3),
    ]
).T
trajectory_example[:, 3] = np.array(
    [
        np.sin(np.linspace(0, 2 * np.pi, time_steps_example) + 4),
        np.cos(np.linspace(0, 2 * np.pi, time_steps_example) + 4),
    ]
).T


radiuss_example = np.array([1, 1, 1, 1])


def animate(i):
    plt.cla()
    axes.set_xlim(-10, 10)
    axes.set_ylim(-10, 10)

    ball1 = plt.Circle(
        trajectory_example[i, 0], radius=radiuss_example[0], color="yellow"
    )
    ball2 = plt.Circle(
        trajectory_example[i, 1], radius=radiuss_example[1], color="blue"
    )
    ball3 = plt.Circle(
        trajectory_example[i, 2], radius=radiuss_example[2], color="green"
    )
    ball4 = plt.Circle(trajectory_example[i, 3], radius=radiuss_example[3], color="red")

    axes.add_patch(ball1)
    axes.add_patch(ball2)
    axes.add_patch(ball3)
    axes.add_patch(ball4)

    plt.plot(
        [trajectory_example[i, 0, 0], trajectory_example[i, 1, 0]],
        [trajectory_example[i, 0, 1], trajectory_example[i, 1, 1]],
        color="black",
    )

    plt.plot(
        [trajectory_example[i, 0, 0], trajectory_example[i, 2, 0]],
        [trajectory_example[i, 0, 1], trajectory_example[i, 2, 1]],
        color="black",
    )

    plt.plot(
        [trajectory_example[i, 0, 0], trajectory_example[i, 3, 0]],
        [trajectory_example[i, 0, 1], trajectory_example[i, 3, 1]],
        color="black",
    )

    plt.plot(
        [trajectory_example[i, 1, 0], trajectory_example[i, 2, 0]],
        [trajectory_example[i, 1, 1], trajectory_example[i, 2, 1]],
        color="black",
    )

    plt.plot(
        [trajectory_example[i, 1, 0], trajectory_example[i, 3, 0]],
        [trajectory_example[i, 1, 1], trajectory_example[i, 3, 1]],
        color="black",
    )

    plt.plot(
        [trajectory_example[i, 2, 0], trajectory_example[i, 3, 0]],
        [trajectory_example[i, 2, 1], trajectory_example[i, 3, 1]],
        color="black",
    )

    return (
        ball1,
        ball2,
        ball3,
        ball4,
    )


# afterwards, switch to zoomable GUI mode

ani = animation.FuncAnimation(
    figure, animate, np.arange(1, time_steps_example), interval=1
)

plt.show()
