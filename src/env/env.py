# what should the trajectory look like?
import numpy as np
from src.bike.bike import Bike


class env:
    def __init__(self, g=-9.8):
        self.trajectory = []
        self.n_bikes = 0
        self.t = 0  # t in seconds
        self.t_step = 0.01  # t step in seconds
        self.bikes = []
        self.starting_positions = np.array([0, 10])  # ???
        self.x_max = 1000
        self.delta_x = 0.001
        self.ground_derivative = 1  ## examplle
        # self.ground = np.array(
        #     [
        #         np.linspace(0, self.x_max, int(self.x_max / self.delta_x)),
        #         np.linspace(0, self.x_max, int(self.x_max / self.delta_x)),
        #     ]
        # ).T  # ground function

        self.ground = np.array(
            [
                np.linspace(0, self.x_max, int(self.x_max / self.delta_x)),
                0 * np.linspace(0, self.x_max, int(self.x_max / self.delta_x)),
            ]
        ).T  # ground function
        self.ms = np.zeros((self.n_bikes, 4))
        self.g = g

    def set_ground(self, ground, derivative):
        self.ground = ground
        self.ground_derivative = derivative

    def get_R(self):
        R = np.zeros((self.n_bikes, 4, 2))
        for i in range(len(self.bikes)):
            R[i, :, :] = self.bikes[i].get_coordinates() + self.starting_positions
        return R

    def get_V(self):
        V = np.zeros((self.n_bikes, 4, 2))
        for i in range(len(self.bikes)):
            V[i, :, :] = self.bikes[i].get_V()
        return V

    def get_K(self):
        K = np.zeros((self.n_bikes, 4, 4))
        for i in range(len(self.bikes)):
            ks = self.bikes[i].get_springs_k()
            K[i, :, :] = ks - np.diag(np.sum(ks, axis=0))

        return K

    def get_B(self):
        B = np.zeros((self.n_bikes, 4, 4))
        for i in range(len(self.bikes)):
            bs = self.bikes[i].get_springs_loss()
            B[i, :, :] = bs - np.diag(np.sum(bs, axis=0))
        return B

    def get_ms(self):
        ms = np.zeros((self.n_bikes, 4))
        for i in range(len(self.bikes)):
            ms[i] = self.bikes[i].get_masses()
        return ms

    def get_torks(self):
        torks = np.zeros((self.n_bikes, 4))
        for i in range(len(self.bikes)):
            torks[i, :2] = self.bikes[i].get_torques()
        return torks

    def get_init_lenghs(self):
        lengths = np.zeros((self.n_bikes, 4, 4))
        for i in range(len(self.bikes)):
            lengths[i, :, :] = self.bikes[i].get_springs_length()

        return lengths

    def get_Radius(self):
        radisuss = np.zeros((self.n_bikes, 2))
        for i in range(self.n_bikes):
            radisuss[i] = self.bikes[i].get_wheels_radius()

        return radisuss

    def set_bikes(self, list_bikes):
        self.bikes = list_bikes
        self.n_bikes = len(list_bikes)
        self.B = self.get_B()
        self.K = self.get_K()
        self.ms = self.get_ms()
        self.init_lengths = self.get_init_lenghs()
        self.torks = self.get_torks()
        self.R = self.get_R()
        self.R0 = self.get_R()
        self.Radiuses = self.get_Radius()

    def get_trajectory_sizes(self):
        return self.trajectory, self.Radiuses

    def evaluate(self):
        delta_X = self.R[:, :, 0] - self.R0[:, :, 0]
        return np.sum(delta_X * self.ms, axis=1) / np.sum(self.ms, axis=1)

    # initalized the whole env and bikes

    def run(self, n):
        self.trajectory = np.zeros((n, self.n_bikes, 4, 2))
        self.t = 0
        self.V = np.zeros((self.n_bikes, 4, 2))

        for i in range(n):
            self.step()
            self.trajectory[i, :, :] = self.R

        self.scores = self.evaluate()

        return self.trajectory, self.scores

    def step(self):
        # ronge kutta method for solving the differential equation

        k1 = self.t_step * self.calculate_acceleration(self.V, self.R, self.t)
        theta1 = self.V

        k2 = self.t_step * self.calculate_acceleration(
            self.V + k1 / 2, self.R + theta1 / 2, self.t + self.t_step / 2
        )
        theta2 = self.V + k1 / 2

        k3 = self.t_step * self.calculate_acceleration(
            self.V + k2 / 2, self.R + theta2 / 2, self.t + self.t_step / 2
        )
        theta3 = self.V + k2 / 2

        k4 = self.t_step * self.calculate_acceleration(
            self.V + k3, self.R + theta3, self.t + self.t_step
        )
        theta4 = self.V + k3

        self.V = self.V + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.R = self.R + (theta1 + 2 * theta2 + 2 * theta3 + theta4) / 6
        self.t += self.t_step

    def calculate_acceleration(self, V, R, t):
        return self.calculate_forces(R, V, t) / np.repeat(
            self.ms.reshape((self.n_bikes, 4, 1)), 2, axis=2
        )

    def cal_gravity_force(self):
        W = np.zeros((self.n_bikes, 4, 2))
        W[:, :, 1] = self.ms * self.g  # shapes = (n_bikes * 4) * 2 = n_bikes * 4 * 2
        return W

    def get_distance_distance_unit_vector(self, R):
        R = R.reshape(self.n_bikes, 4, 1, 2)
        tiled = np.tile(R, (1, 1, 4, 1))

        transposed = tiled.transpose((0, 2, 1, 3))

        distance = transposed - tiled

        normalized_distances = distance / np.linalg.norm(distance, axis=3).reshape(
            -1, 4, 4, 1
        )
        normalized_distances[:, range(4), range(4)] = 0

        return distance, normalized_distances

    def cal_spring_force(self, R):  ### values not tested
        d, d_hat = self.get_distance_distance_unit_vector(R)

        x = d - (d_hat * self.init_lengths.reshape(-1, 4, 4, 1))  # -/+ ?

        spring_force = np.sum(-self.K.reshape(-1, 4, 4, 1) * x, axis=1)  # or axis = 1,2

        return spring_force

    def cal_damping_force(self, V):
        return -self.B @ V

    # checked functions above

    def calculate_distance_from_ground(self, pos):

        norm = np.linalg.norm(
            np.tile(self.ground, (3, 1, 1)).transpose((1, 0, 2)) - pos, axis=2
        )

        return np.min(norm, axis=0), np.argmin(norm, axis=0)

    def calculate_forces(self, R, V, t):
        force = np.zeros((self.n_bikes, 4, 2))

        W = self.cal_gravity_force()
        force += W
        spring_force = self.cal_spring_force(R)
        force += spring_force
        damping_force = self.cal_damping_force(V)
        force += damping_force

        # n_hat = self.perpendicular_unit_vector(R)  # shape = n_bikes * 4 * 2
        # t_hat = self.parallel_unit_vector(R)  # shape = n_bikes * 4 * 2
        # C = self.get_connected(R)  # shape = n_bikes * 4

        # turk_force = (
        #     self.torks * C @ t_hat
        # )  # shape = (n_bikes * 4) * (n_bikes * 4) * (n_bikes * 4 * 2) = n_bikes * 4 * 2   ?????

        # # M * X_vec.. = - K (X_vec - l0 * d_hat) - B * V_vec + W_vec + tork * T * C + N * C

        # Force = W + damping_force + spring_force + turk_force
        # Force -= (
        #     np.dot(Force, n_hat, axis=...) * n_hat
        # )  # remove the perpendicular component
        return force

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

    def get_connected(self, R):
        radiuses = self.get_Radius()

        C = np.zeros((self.n_bikes, 4), dtype=bool)
        C[:, 0] = self.ground(R[:, 0, 0]) > R[:, 0, 1]
        C[:, 1] = self.ground(R[:, 1, 0]) > R[:, 1, 1]
        distance_c = self.calculate_distance_from_ground(R[:, 2])
        distance_d = self.calculate_distance_from_ground(R[:, 3])

        C[:, 2] = (
            np.abs(distance_c - radiuses[:, 0]) < 0e-5
        )  # or almost zero ????   it should be compared to the radius of the wheel
        C[:, 3] = (
            np.abs(distance_d - radiuses[:, 1]) < 0e-5
        )  # or almost zero ???/ it should be compared to the radius of the wheel

        return C
