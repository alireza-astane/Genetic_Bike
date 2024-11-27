from bike.bike import Bike
from env.env import env
import numpy as np
from pprint import pprint

my_env = env(9.8)


my_bike_1 = Bike(
    1, 1, 2, 3, 1, 4, 1, 2, 3, 0, 1, 3, 2, 4, 3, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
)

my_bike_2 = Bike(
    1, 1, 2, 3, 1, 4, 1, 2, 3, 0, 1, 3, 2, 4, 3, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
)

my_bike_3 = Bike(
    1, 1, 2, 3, 1, 4, 1, 2, 3, 0, 1, 3, 2, 4, 3, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
)

my_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])


print(my_env.get_distance_unit_vector(my_env.R))
