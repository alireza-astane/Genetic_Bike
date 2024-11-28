from bike.bike import Bike
from env.env import env
import numpy as np
from pprint import pprint
from visualiation.vis import Vis
from evolution.evo import GeneticAlgorithm

my_env = env(-9.8)

    # Genetic Algorithm parameters
population_size = 10
num_bits_per_individual = 28  # Adjust based on your scale and precision needs
num_generations = 100
crossover_probability = 0.7
mutation_probability = 0.05
num_parents = 4
tolerance = 0.01

# Create Genetic Algorithm instance
ga = GeneticAlgorithm(
    populationSize=population_size,
    numBitsPerIndividual=num_bits_per_individual,
    numGenerations=num_generations,
    crossoverProbability=crossover_probability,
    mutationProbability=mutation_probability,
    numParents=num_parents,
    fitnessFunction=None,
    tolerance=tolerance
)


decoded_params =  np.zeros(28)

# Create some dummy bikes (this step might be different depending on your actual Bike class implementation)
for _ in range(population_size):
    bike = Bike(
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
    10
    )
    ga.bikes.append(bike)
    ga.bike2array(len(ga.bikes) - 1, bike)



my_env.set_bikes(ga.bikes)

# Run the genetic algorithm
ga.fit(my_env)

# steps = 20

# trajectory, scores = my_env.run(steps)

# print(scores)

# trajectory, sizes = my_env.get_trajectory_sizes()



# my_vis = Vis(trajectory[:, 0], sizes[0], steps, my_env.ground)

# my_vis.run()
