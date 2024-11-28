# add all fields to init
# add doc
# add tests
# fix evaluate
import numpy as np
from src.bike.bike import Bike
from tqdm import tqdm


class env:
    def __init__(
        self,
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    ):
        self.elasticiity = elasticiity
        self.trajectory = []
        self.n_bikes = 0
        self.t = 0
        self.t_step = t_step  # t step in seconds
        self.bikes = []
        self.starting_positions = np.array([0, starting_hight])
        self.x_max = x_max
        self.delta_x = delta_x
        self.ground_derivative = ground_derivative
        self.ground = ground
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
        torks = np.zeros((self.n_bikes, 2))
        for i in range(len(self.bikes)):
            torks[i, :] = self.bikes[i].get_torques()
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
        ys = self.trajectory[:, :, 2:, 1]
        y_ground = self.ground[
            (self.trajectory[:, :, 2:, 0] / self.delta_x).astype(np.int64), 1
        ]
        mass_center_change = np.sum(delta_X * self.ms, axis=1) / np.sum(self.ms, axis=1)
        lower = ys < y_ground
        failed = np.any(lower, axis=(0, 2))
        # where_failed = np.argwhere(lower)
        # wheels_x = self.trajectory[:, :, 2:, 0]
        # print(where_failed, where_failed.shape, wheels_x.shape)

        # wheels_x[where_failed[:, 0], where_failed[:, 1], where_failed[:, 2]].shape

        mass_center_change[failed] = 0

        return mass_center_change

    def run(self, n):
        self.trajectory = np.zeros((n, self.n_bikes, 4, 2))
        self.t = 0
        self.V = np.zeros((self.n_bikes, 4, 2))

        for i in tqdm(range(n)):
            self.step()
            self.trajectory[i, :, :] = self.R

        self.scores = self.evaluate()

        return self.trajectory, self.scores

    def step(self):
        # ronge kutta method for solving the differential equation
        a1, n_hat, is_touched = self.calculate_acceleration(self.V, self.R, self.t)
        k1 = self.t_step * a1
        k1 = self.apply_touching_effect(k1, n_hat, is_touched)
        theta1 = self.V

        a2, n_hat, is_touched = self.calculate_acceleration(
            self.V + k1 / 2, self.R + theta1 / 2, self.t + self.t_step / 2
        )
        k2 = self.t_step * a2
        k2 = self.apply_touching_effect(k2, n_hat, is_touched)
        theta2 = self.V + k1 / 2

        a3, n_hat, is_touched = self.calculate_acceleration(
            self.V + k2 / 2, self.R + theta2 / 2, self.t + self.t_step / 2
        )
        k3 = self.t_step * a3
        k3 = self.apply_touching_effect(k3, n_hat, is_touched)
        theta3 = self.V + k2 / 2

        a4, n_hat, is_touched = self.calculate_acceleration(
            self.V + k3, self.R + theta3, self.t + self.t_step
        )
        k4 = self.t_step * a4
        k4 = self.apply_touching_effect(k4, n_hat, is_touched)
        theta4 = self.V + k3

        self.V = self.V + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.V = self.apply_touching_effect(self.V, n_hat, is_touched)
        self.R = self.R + (theta1 + 2 * theta2 + 2 * theta3 + theta4) / 6
        self.t += self.t_step

    def apply_touching_effect(self, K, n_hat, is_touched):
        dv_n = n_hat * is_touched.reshape((self.n_bikes, 2, 1)) * (1 + self.elasticiity)

        V_dot_n_hat = np.sum(K[:, :2, :] * n_hat, axis=2).reshape((self.n_bikes, 2, 1))

        K[:, :2, :] -= V_dot_n_hat * dv_n

        return K

    def calculate_acceleration(self, V, R, t):
        force, n_hat, is_touched = self.calculate_forces(R, V, t)
        return (
            force / np.repeat(self.ms.reshape((self.n_bikes, 4, 1)), 2, axis=2),
            n_hat,
            is_touched,
        )

    def cal_gravity_force(self):
        W = np.zeros((self.n_bikes, 4, 2))
        W[:, :, 1] = self.ms * self.g
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

    def cal_spring_force(self, R):
        d, d_hat = self.get_distance_distance_unit_vector(R)

        x = d - (d_hat * self.init_lengths.reshape(-1, 4, 4, 1))

        spring_force = np.sum(-self.K.reshape(-1, 4, 4, 1) * x, axis=1)

        return spring_force

    def cal_damping_force(self, V):
        return self.B @ V

    def perpendicular_unit_vector(self, touch_point_x):
        n_hat = np.zeros((self.n_bikes, 2, 2))

        f_primes = self.ground_derivative[touch_point_x][:, :, 1]

        n_hat[:, :, 0] = -f_primes / (1 + f_primes**2) ** 0.5
        n_hat[:, :, 1] = 1 / (1 + f_primes**2) ** 0.5

        return n_hat

    def parallel_unit_vector(self, touch_point_x):
        t_hat = np.zeros((self.n_bikes, 2, 2))

        f_primes = self.ground_derivative[touch_point_x][:, :, 1]

        t_hat[:, :, 0] = 1 / (1 + f_primes**2) ** 0.5
        t_hat[:, :, 1] = f_primes / (1 + f_primes**2) ** 0.5

        return t_hat

    def get_connection_info(self, pos):

        tiled_ground = np.tile(self.ground, (self.n_bikes, 2, 1, 1)).transpose(
            (2, 0, 1, 3)
        )

        norm_distacnes_from_ground = np.linalg.norm(tiled_ground - pos, axis=3)

        min_distacne = np.min(norm_distacnes_from_ground, axis=0)

        closest_point_x = np.argmin(norm_distacnes_from_ground, axis=0)

        is_touched = min_distacne - self.Radiuses < 1e-5

        return min_distacne, closest_point_x, is_touched

    def calculate_forces(self, R, V, t):
        force = np.zeros((self.n_bikes, 4, 2))

        W = self.cal_gravity_force()
        spring_force = self.cal_spring_force(R)
        damping_force = self.cal_damping_force(V)

        min_distacne, closest_point_x, is_touched = self.get_connection_info(
            R[:, :2, :]
        )

        n_hat = self.perpendicular_unit_vector(closest_point_x)
        t_hat = self.parallel_unit_vector(closest_point_x)

        turk_force = np.zeros((self.n_bikes, 4, 2))

        turk_force[:, :2, :] = t_hat * (self.torks * is_touched).reshape(
            (self.n_bikes, 2, 1)
        )

        force += W + turk_force + spring_force + damping_force

        return force, n_hat, is_touched


## check if the masses touched ground in evaluate function
