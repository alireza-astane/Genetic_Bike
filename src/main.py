from bike.bike import Bike
from env.env import Environment
import numpy as np
from visualiation.vis import Vis
from evolution.evo import GeneticAlgorithm
import random

# Environments parameters
steps = 100
my_env = Environment(-9.8)

bike = Bike(
    0,
    0,
    1,
    1,
    1,
    10,
    0,
    1,
    1,
    0,
    0,
    2,
    1,
    10,
    2,
    1,
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

my_env.set_bikes([bike])
trajectory, scores = my_env.run(steps)

print(trajectory.shape)

trajectory, sizes = my_env.get_trajectory_sizes()

my_vis = Vis(trajectory[:, 0], sizes[0], steps, my_env.ground)

my_vis.run()

# # Genetic Algorithm parameters
# population_size = 3
# num_bits_per_individual = 28  # Adjust based on your scale and precision needs
# num_generations = 10
# crossover_probability = 0.7
# mutation_probability = 0.05
# num_parents = 2
# tolerance = 0.01


# # Create Genetic Algorithm instance
# ga = GeneticAlgorithm(
#     populationSize=population_size,
#     numBitsPerIndividual=num_bits_per_individual,
#     numGenerations=num_generations,
#     crossoverProbability=crossover_probability,
#     mutationProbability=mutation_probability,
#     numParents=num_parents,
#     fitnessFunction=None,
#     tolerance=tolerance,
# )

# for _ in range(population_size):
#     bike = Bike(

#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10), random.randint(1, 10), random.randint(1, 10),
#         random.randint(1, 10)

#     )
#     ga.bikes.append(bike)
#     ga.bike2array(len(ga.bikes) - 1, bike)


# # Create some dummy bikes (this step might be different depending on your actual Bike class implementation)
# # for _ in range(population_size):
# #     bike = Bike(

# #     1 + 0.2 * np.random.rand(),
# #     1 + 0.2 * np.random.rand(),
# #     2 + 0.2 * np.random.rand(),
# #     3 + 0.2 * np.random.rand(),
# #     1 + 0.2 * np.random.rand(),
# #     4 + 0.2 * np.random.rand(),
# #     1 + 0.2 * np.random.rand(),
# #     2 + 0.2 * np.random.rand(),
# #     3 + 0.2 * np.random.rand(),
# #     0 + 0.2 * np.random.rand(),
# #     1 + 0.2 * np.random.rand(),
# #     3 + 0.2 * np.random.rand(),
# #     2 + 0.2 * np.random.rand(),
# #     4 + 0.2 * np.random.rand(),
# #     3 + 0.2 * np.random.rand(),
# #     2 + 0.2 * np.random.rand(),
# #     100 + 2 * np.random.rand(),
# #     100 + 2 * np.random.rand(),
# #     100 + 2 * np.random.rand(),
# #     100 + 2 * np.random.rand(),
# #     100 + 2 * np.random.rand(),
# #     100 + 2 * np.random.rand(),
# #     10 + 1 * np.random.rand(),
# #     10 + 1 * np.random.rand(),
# #     10 + 1 * np.random.rand(),
# #     10 + 1 * np.random.rand(),
# #     10 + 1 * np.random.rand(),
# #     10 + 1 * np.random.rand(),

# #     )
# #     ga.bikes.append(bike)
# #     ga.bike2array(len(ga.bikes) - 1, bike)

# my_env.set_bikes(ga.bikes)

# print("Initial population:")
# print(ga.population)

# # Calculate fitness


# trajectory, scores = my_env.run(steps)
# ga.populationFitness = scores
# print("Initial fitness:")
# print(ga.populationFitness)

# for i in range(ga.numGenerations):
#     print(f"\nGeneration {i + 1}:")

#     ga.elites[i] = ga.population[np.argmax(ga.populationFitness)].copy()
#     print("Elites:")
#     print(ga.elites[i])

#     ga.selection()
#     print("Selected Parents:")
#     print(ga.parents)

#     ga.crossover()
#     print("Population after crossing:")
#     print(ga.population)

#     ga.mutation()
#     print("Population after mutation:")
#     print(ga.population)
#     # Update fitness
#     ga.population[np.argmin(ga.populationFitness)] = ga.elites[i].copy()

#     ga.calculateStatistics()

#     trajectory, scores = my_env.run(steps)
#     ga.populationFitness = scores

#     print("Generation Fitness:")
#     print(ga.populationFitness)

#     if ga.tolerance < ga.maxValues[-1]:
#         print("Last iteration", i)
#         ga.elites[i] = ga.population[np.argmax(ga.populationFitness)]
#         break


# print(scores)

# trajectory, sizes = my_env.get_trajectory_sizes()

# my_vis = Vis(trajectory[:, 0], sizes[0], steps, my_env.ground)

# my_vis.run()
