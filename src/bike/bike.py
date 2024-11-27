import numpy as np


class Bike:

    def __init__(self, 
        wheel_1_x, wheel_1_y, wheel_1_radius, wheel_1_mass, wheel_1_torque,
        wheel_2_x, wheel_2_y, wheel_2_radius, wheel_2_mass, wheel_2_torque,
        body_1_x, body_1_y, body_1_mass,
        body_2_x, body_2_y, body_2_mass,
        k_spring_w1_w2, k_spring_w1_b1, k_spring_w1_b2, 
        k_spring_w2_b1, k_spring_w2_b2,
        k_spring_b1_b2,
        loss_spring_w1_w2, loss_spring_w1_b1, loss_spring_w1_b2, 
        loss_spring_w2_b1, loss_spring_w2_b2,
        loss_spring_b1_b2):

        self.wheel_1_x      = wheel_1_x
        self.wheel_1_y      = wheel_1_y
        self.wheel_1_radius = wheel_1_radius
        self.wheel_1_mass   = wheel_1_mass
        self.wheel_1_torque = wheel_1_torque

        self.wheel_2_x      = wheel_2_x
        self.wheel_2_y      = wheel_2_y
        self.wheel_2_radius = wheel_2_radius
        self.wheel_2_mass   = wheel_2_mass
        self.wheel_2_torque = wheel_2_torque

        self.body_1_x       = body_1_x
        self.body_1_y       = body_1_y
        self.body_1_mass    = body_1_mass

        self.body_2_x       = body_2_x
        self.body_2_y       = body_2_y
        self.body_2_mass    = body_2_mass

        self.k_spring_w1_w2 = k_spring_w1_w2
        self.k_spring_w1_b1 = k_spring_w1_b1
        self.k_spring_w1_b2 = k_spring_w1_b2
        self.k_spring_w2_b1 = k_spring_w2_b1
        self.k_spring_w2_b2 = k_spring_w2_b2
        self.k_spring_b1_b2 = k_spring_b1_b2

        self.loss_spring_w1_w2 = loss_spring_w1_w2
        self.loss_spring_w1_b1 = loss_spring_w1_b1
        self.loss_spring_w1_b2 = loss_spring_w1_b2
        self.loss_spring_w2_b1 = loss_spring_w2_b1
        self.loss_spring_w2_b2 = loss_spring_w2_b2
        self.loss_spring_b1_b2 = loss_spring_b1_b2


    def get_coordinates(self):
        nodes_coordinates = np.array([
            [self.wheel_1_x, self.wheel_1_y],
            [self.wheel_2_x, self.wheel_2_y],
            [self.body_1_x, self.body_1_y],
            [self.body_2_x, self.body_2_y]
            ])

        return nodes_coordinates

    def get_springs_k(self):
        springs_k = np.array([        
            [0, self.k_spring_w1_w2, self.k_spring_w1_b1, self.k_spring_w1_b2],
            [self.k_spring_w1_w2, 0, self.k_spring_w2_b1, self.k_spring_w2_b2],
            [self.k_spring_w1_b1, self.k_spring_w2_b1, 0, self.k_spring_b1_b2],
            [self.k_spring_w1_b2, self.k_spring_w2_b2, self.k_spring_b1_b2, 0]])

        return springs_k

    def get_springs_loss(self):
        springs_loss = np.array([
            [0, self.loss_spring_w1_w2, self.loss_spring_w1_b1, self.loss_spring_w1_b2],
            [self.loss_spring_w1_w2, 0, self.loss_spring_w2_b1, self.loss_spring_w2_b2],
            [self.loss_spring_w1_b1, self.loss_spring_w2_b1, 0, self.loss_spring_b1_b2],
            [self.loss_spring_w1_b2, self.loss_spring_w2_b2, self.loss_spring_b1_b2, 0]])

        return springs_loss

    def get_torques(self):

        return [self.wheel_1_torque, self.wheel_1_torque]

    def spring_length(self, node_i, node_j):
        
        nodes_coordinates = np.array([
            [self.wheel_1_x, self.wheel_1_y],
            [self.wheel_2_x, self.wheel_2_y],
            [self.body_1_x, self.body_1_y],
            [self.body_2_x, self.body_2_y]
            ])

        x_1 = nodes_coordinates[node_i][0]
        y_1 = nodes_coordinates[node_i][1]

        x_2 = nodes_coordinates[node_j][0]
        y_2 = nodes_coordinates[node_j][1]

        delta_x = x_1 - x_2
        delta_y = y_1 - y_2

        my_spring_length = (delta_x**2 + delta_y**2)**0.5

        return my_spring_length

    def get_springs_length(self):
        springs_length = np.zeros((4,4))
        for i in range(4):
            for j in range(4):

                if i==j: 
                    springs_length[i][j] = 0
                else:
                    springs_length[i][j] = self.spring_length(i, j)

        return springs_length

    def get_masses(self):

        return [self.wheel_1_mass, self.wheel_2_mass, self.body_1_mass, self.body_2_mass]
