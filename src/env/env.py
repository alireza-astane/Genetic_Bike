# what should the trajectory look like?


class env:
    def __init__(self, g):
        self.trajectory = []
        self.n_bikes = 0
        self.time = 0  # time in seconds
        self.time_step = 1  # time step in seconds
        self.bikes = []
        self.starting_positions = np.zeros((self.n_bikes, 2))
        self.ground_derivative = None
        self.ground = None
        self.ms = np.zeros((self.n_bikes))
        self.g = g

    def set_bikes(self, list_bikes):
        self.bikes = list_bikes

    def get_trajectory(self):
        # maybe someone wants to get the trajectory to visualize it
        return self.trajectory

    def evaluate(self, R0, R):

        # evaluate the current state of the environment

        # compare last element of trajectory with the starting positions

        scores = R[:, 0] = R0[:, 0]
        return scores

    def run(self, n):
        trajectory = np.zeros((n, self.n_bikes, 2))
        # run the environment
        self.time = 0
        R, V, t = ...
        R0 = R
        for i in range(n):
            R, V, t = self.step(R, V, t, self.calculate_acceleration)
            self.time += self.time_step
            trajectory[i, :, :] = R

        scores = self.evaluate(R0, R)

        return trajectory, scores

    def create_ground(self):
        # example
        self.ground = np.sin
        self.ground_derivative = np.cos

    def step(self, R, V, t, acceleration_function):
        # ronge kutta method for solving the differential equation

        k1 = self.time_step * acceleration_function(V, R, t)
        theta1 = V

        k2 = self.time_step * acceleration_function(
            V + k1 / 2, R + theta1 / 2, t + self.time_step / 2
        )
        theta2 = V + k1 / 2

        k3 = self.time_step * acceleration_function(
            V + k2 / 2, R + theta2 / 2, t + self.time_step / 2
        )
        theta3 = V + k2 / 2

        k4 = self.time_step * acceleration_function(
            V + k3, R + theta3, t + self.time_step
        )
        theta4 = V + k3

        V = V + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        R = R + (theta1 + 2 * theta2 + 2 * theta3 + theta4) / 6

        return R, V, t

    def calculate_acceleration(self, R, V, t):
        return self.calculate_forces(R, V, t) / self.ms

    def calculate_forces(self, R, V, t):
        # calculate the forces acting on the bike based on the trajectory and env    #### ???????/
        pass

    def perpendicular_unit_vector(self, R, V, t):
        # n_bikes*4*2
        n_hat = np.zeros((self.n_bikes, 4, 2))
        # n_hat[:,:2,:] = 0 no force to top massses
        n_hat[:, 2, 0] = (
            -self.ground_derivative(R[:, 2, 0])
            / (1 + self.ground_derivative(R[:, 2, 0]) ** 2) ** 0.5
        )
        n_hat[:, 2, 1] = 1 / (1 + self.ground_derivative(R[:, 2, 0]) ** 2) ** 0.5
        n_hat[:, 3, 0] = (
            -self.ground_derivative(R[:, 3, 0])
            / (1 + self.ground_derivative(R[:, 3, 0]) ** 2) ** 0.5
        )
        n_hat[:, 3, 1] = 1 / (1 + self.ground_derivative(R[:, 3, 0]) ** 2) ** 0.5

        return n_hat

    def parallel_unit_vector(self, R, V, t):
        t_hat = np.zeros((self.n_bikes, 4, 2))
        t_hat[:, 2, 0] = 1 / (1 + self.ground_derivative(R[:, 2, 0]) ** 2) ** 0.5
        t_hat[:, 2, 1] = (
            self.ground_derivative(R[:, 2, 0])
            / (1 + self.ground_derivative(R[:, 2, 0]) ** 2) ** 0.5
        )

        t_hat[:, 3, 0] = 1 / (1 + self.ground_derivative(R[:, 3, 0]) ** 2) ** 0.5
        t_hat[:, 3, 1] = (
            self.ground_derivative(R[:, 3, 0])
            / (1 + self.ground_derivative(R[:, 3, 0]) ** 2) ** 0.5
        )

        return t_hat

    def weight(self, masses):
        w = np.zeros((self.n_bikes, 4, 2))
        w[:, :, 1] = -masses * self.g
        return w

    def turk(self, bike):
        # calculate the torque on the bike  ????????
        return np.zeros((self.n_bikes, 2))
