class Bike:

    def __init__(self, 
        wheel_1_x, wheel_1_y, wheel_1_radius, wheel_1_mass, wheel_1_torque=0,
        wheel_2_x, wheel_2_y, wheel_2_radius, wheel_2_mass, wheel_2_torque=0,
        body_1_x, body_1_y, body_1_mass,
        body_2_x, body_2_y, body_2_mass,):

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

    nodes_coordinates = [
        [self.wheel_1_x, self.wheel_1_y],
        [self.wheel_2_x, self.wheel_2_y],
        [self.body_1_x, self.body_1_y],
        [self.body_2_x, self.body_2_y]
        ]

    def get_coordinates():
        
        return nodes_coordinates

    def springs_i_j(self, node_i, node_j, spring_coefficient, loss_coefficient):
        
        x_1 = nodes_coordinates[node_i][0]
        y_1 = nodes_coordinates[node_i][1]

        x_2 = nodes_coordinates[node_j][0]
        y_2 = nodes_coordinates[node_j][1]

        delta_x = x_1 - x_2
        delta_y = y_1 - y_2

        spring_length = (delta_x**2 + delta_y**2)**0.5

        return [spring_length, spring_coefficient, loss_coefficient]
