# what should the trajectory look like?


class env:
    def __init__(self, g):
        self.trajectory = []
        self.n_bikes = 0
        self.time = 0  # time in seconds
        self.time_step = 1  # time step in seconds
        self.bikes = []
        self.starting_positions = np.array([0, 10])  # ???
        self.ground_derivative = None
        self.ground = None
        self.ms = np.zeros((self.n_bikes, r))
        self.g = -g * np.array([0, 1])

    def set_bikes(self, list_bikes):
        self.bikes = list_bikes

    def get_R(self):
        R = np.zeros((self.n_bikes, 4, 2))
        for i in self.bikes:
            R[i, :, :] = self.bikes[i].get_R() + self.starting_positions
        return R

    def get_V(self):
        V = np.zeros((self.n_bikes, 4, 2))
        for i in self.bikes:
            V[i, :, :] = self.bikes[i].get_V()
        return V

    def get_K(self):
        K = np.zeros((self.n_bikes, 4, 4))
        for i in self.bikes:
            K[i, :, :] = self.bikes[i].get_K()
        return K

    def get_B(self):
        B = np.zeros((self.n_bikes, 4, 4))
        for i in self.bikes:
            B[i, :, :] = self.bikes[i].get_B()
        return B

    def get_ms(self):
        ms = np.zeros((self.n_bikes, 4))
        for i in self.bikes:
            ms[i] = self.bikes[i].get_m()
        return ms

    def get_torks(self):
        torks = np.zeros((self.n_bikes, 4))
        for i in self.bikes:
            torks[i, :] = self.bikes[i].get_tork()
        return torks

    def get_init_lenghs(self):
        lengths = np.zeros((self.n_bikes, 4, 4))
        for i in self.bikes:
            lengths[i, :, :] = self.bikes[i].get_init_lengths()

        return lengths

    def get_trajectory(self):
        return self.trajectory

    def evaluate(self):
        return self.R[:, 0] - self.R0[:, 0]

    def run(self, n):
        self.trajectory = np.zeros((n, self.n_bikes, 2))
        self.time = 0
        self.R, self.V = self.get_R(), self.get_V()
        self.K = self.get_K()
        self.B = self.get_B()
        self.ms = self.get_ms()
        self.torks = self.get_torks()
        self.init_lengths = self.get_init_lenghs()

        self.R0 = R
        for i in range(n):
            self.step(R, V, t, self.calculate_acceleration)
            self.time += self.time_step
            trajectory[i, :, :] = R

        self.scores = self.evaluate(R0, R)

        return self.trajectory, self.scores

    def create_ground(self, func=np.sin, derivative=np.cos):
        self.ground = func
        self.ground_derivative = derivative

        return self.ground, self.ground_derivative

    def step(self):
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

    def perpendicular_unit_vector(self, R):
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

    def parallel_unit_vector(self, R):
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

    def get_distance_unit_vector(self, R):
        distance = R - np.traspose(...)
        normalized_distance = distance / np.linalg.norm(distance, axis=2)
        # put zeros for diagnal elements
        normalized_distance[range(self.n_bikes), range(self.n_bikes)] = 0  # ?????
        return normalized_distance

    def calculate_forces(self, R, V, t):
        # calculate the forces acting on the bike based on the trajectory and env    #### ???????/
        pass
        W = self.ms * self.g  # shapes = (n_bikes * 4) * 2 = n_bikes * 4 * 2
        damping_force = (
            -self.B @ V
        )  # shape = (n_bikes * 4 * 4 ) @ (n_bikes * 4 * 2 ) = n_bikes * 4 * 2
        initial_lengths = self.init_lengths @ self.get_distance_unit_vector(
            R
        )  # shape = (n_bikes * 4 * 4) @ (n_bikes * 4 * 2) = n_bikes * 4 * 2
        spring_force = -self.K @ (
            R - self.initial_lengths
        )  # shape = (n_bikes * 4 * 4) @ (n_bikes * 4 * 2) = n_bikes * 4 * 2
        n_hat = self.perpendicular_unit_vector(R)  # shape = n_bikes * 4 * 2
        t_hat = self.parallel_unit_vector(R)  # shape = n_bikes * 4 * 2
        C = self.get_connected(R)  # shape = n_bikes * 4

        turk_force = (
            self.torks * C @ t_hat
        )  # shape = (n_bikes * 4) * (n_bikes * 4) * (n_bikes * 4 * 2) = n_bikes * 4 * 2   ?????

        # M * X_vec.. = - K (X_vec - l0 * d_hat) - B * V_vec + W_vec + tork * T * C + N * C

        Force = W + damping_force + spring_force + turk_force
        Force -= (
            np.dot(Force, n_hat, axis=...) * n_hat
        )  # remove the perpendicular component
        return Force

    def get_connected(self, R):
        C = np.zeros((self.n_bikes, 4), dtype=bool)
        C[:, 0] = self.ground(R[:, 0, 0]) > R[:, 0, 1]
        C[:, 1] = self.ground(R[:, 1, 0]) > R[:, 1, 1]
        distance_c = self.calculate_distance_from_ground(R[:, 2])
        distance_d = self.calculate_distance_from_ground(R[:, 3])

        C[:, 2] = distance_c == 0  # or almost zero ????
        C[:, 3] = distance_d == 0  # or almost zero ???/

        return C

    def calculate_distance_from_ground(self, pos):
        a = pos[:, 0]
        b = -self.ground_derivative(pos[:, 0]) * (pos[:, 1] - self.ground(pos[:, 0]))

        min_xs = np.min(
            np.array([a / 2 + np.sqrt(a**2 / 4 + b), a / 2 - np.sqrt(a**2 / 4 + b)]),
            axis=1,
        )

        return self.get_distance_from_x(pos, min_xs)

    def get_distance_from_x(self, pos, x):
        return np.sqrt((pos[:, 0] - x) ** 2 + (pos[:, 1] + self.groun(x)) ** 2)


### make sum tests and run the code
