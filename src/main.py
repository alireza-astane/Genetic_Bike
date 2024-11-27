from bike.bike import Bike
from env.env import env
import numpy as np
from pprint import pprint
from visualiation.vis import Vis


my_env = env(9.8)


my_bike_1 = Bike(
    1, 2, 1, 3, 1, 4, 1, 2, 3, 0, 1, 3, 2, 4, 3, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
)

my_bike_2 = Bike(
    1, 1, 2, 3, 1, 4, 1, 2, 3, 0, 1, 3, 2, 4, 3, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
)

my_bike_3 = Bike(
    1, 1, 2, 3, 1, 4, 1, 2, 3, 0, 1, 3, 2, 4, 3, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
)

my_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])


my_env.run(1000)

trajectory, sizes = my_env.get_trajectory_sizes()


my_vis = Vis(trajectory[:, 0], sizes[0], 1000)

my_vis.run()
