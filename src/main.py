from bike.bike import Bike
from env.env import env
import numpy as np
from pprint import pprint
from visualiation.vis import Vis
from evolution.evo import GeneticAlgorithm
import random

# Environments parameters
steps = 20
my_env = env(-9.8)


my_bike_1 = Bike(
    1,
    1,
    2,
    3,
    1,
    4,
    1,
    2,
    3,
    0,
    1,
    3,
    2,
    4,
    3,
    2,
    100,
    100,
    100,
    100,
    100,
    100,
    10,
    10,
    10,
    10,
    10,
    10,
)

my_bike_2 = Bike(
    1,
    1,
    2,
    3,
    1,
    4,
    1,
    2,
    3,
    0,
    1,
    3,
    2,
    4,
    3,
    2,
    100,
    100,
    100,
    100,
    100,
    100,
    10,
    10,
    10,
    10,
    10,
    10,
)

my_bike_3 = Bike(
    1,
    1,
    2,
    3,
    1,
    4,
    1,
    2,
    3,
    0,
    1,
    3,
    2,
    4,
    3,
    2,
    100,
    100,
    100,
    100,
    100,
    100,
    10,
    10,
    10,
    10,
    10,
    10,
)


my_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])

# my_env.set_bikes([my_bike_1])

steps = 20

trajectory, scores = my_env.run(steps)
ga.populationFitness = scores
print("Initial fitness:")
print(ga.populationFitness)

for i in range(ga.numGenerations):
    print(f"\nGeneration {i + 1}:")

    ga.elites[i] = ga.population[np.argmax(ga.populationFitness)].copy()
    print("Elites:")
    print(ga.elites[i])

    ga.selection()
    print("Selected Parents:")
    print(ga.parents)

    ga.crossover()
    print("Population after crossing:")
    print(ga.population)

    ga.mutation()
    print("Population after mutation:")
    print(ga.population)
    # Update fitness
    ga.population[np.argmin(ga.populationFitness)] = ga.elites[i].copy()

    ga.calculateStatistics()

    trajectory, scores = my_env.run(steps)
    ga.populationFitness = scores

    print("Generation Fitness:")
    print(ga.populationFitness)

    if ga.tolerance < ga.maxValues[-1]:
        print("Last iteration", i)
        ga.elites[i] = ga.population[np.argmax(ga.populationFitness)]
        break


print(scores)

trajectory, sizes = my_env.get_trajectory_sizes()

my_vis = Vis(trajectory[:, 0], sizes[0], steps, my_env.ground)

my_vis.run()
