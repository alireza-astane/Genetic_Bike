from evo import GeneticAlgorithm
from bike import Bike
import numpy as np
import random
import matplotlib.pyplot as plt


def absolute_parameters_fitness(ga_instance, chromosome):
    """
    Fitness function to minimize the sum of absolute values of bike parameters.
    The goal is to get as close to 0 for all parameters as possible.
    """
    # Convert binary chromosome to decimal values
    decoded_params = ga_instance.binarytobike(chromosome)

    # Create a Bike instance with the decoded parameters
    bike = Bike(
        decoded_params[0],
        decoded_params[1],
        decoded_params[2],
        decoded_params[3],
        decoded_params[4],
        decoded_params[5],
        decoded_params[6],
        decoded_params[7],
        decoded_params[8],
        decoded_params[9],
        decoded_params[10],
        decoded_params[11],
        decoded_params[12],
        decoded_params[13],
        decoded_params[14],
        decoded_params[15],
        decoded_params[16],
        decoded_params[17],
        decoded_params[18],
        decoded_params[19],
        decoded_params[20],
        decoded_params[21],
        decoded_params[22],
        decoded_params[23],
        decoded_params[24],
        decoded_params[25],
        decoded_params[26],
        decoded_params[27],
    )

    # Calculate sum of absolute values
    total_abs_sum = (
        abs(bike.wheel_1_x)
        + abs(bike.wheel_1_y)
        + abs(bike.wheel_1_radius)
        + abs(bike.wheel_1_mass)
        + abs(bike.wheel_1_torque)
        + abs(bike.wheel_2_x)
        + abs(bike.wheel_2_y)
        + abs(bike.wheel_2_radius)
        + abs(bike.wheel_2_mass)
        + abs(bike.wheel_2_torque)
        + abs(bike.body_1_x)
        + abs(bike.body_1_y)
        + abs(bike.body_1_mass)
        + abs(bike.body_2_x)
        + abs(bike.body_2_y)
        + abs(bike.body_2_mass)
        + abs(bike.k_spring_w1_w2)
        + abs(bike.k_spring_w1_b1)
        + abs(bike.k_spring_w1_b2)
        + abs(bike.k_spring_w2_b1)
        + abs(bike.k_spring_w2_b2)
        + abs(bike.k_spring_b1_b2)
        + abs(bike.loss_spring_w1_w2)
        + abs(bike.loss_spring_w1_b1)
        + abs(bike.loss_spring_w1_b2)
        + abs(bike.loss_spring_w2_b1)
        + abs(bike.loss_spring_w2_b2)
        + abs(bike.loss_spring_b1_b2)
    )

    # The fitness is the negative of the sum (so minimizing absolute values maximizes fitness)
    return -total_abs_sum


def test_geneticAlgorithm():
    # Genetic Algorithm parameters
    population_size = 50
    num_bits_per_individual = 28  # Adjust based on your scale and precision needs
    num_generations = 100
    crossover_probability = 0.7
    mutation_probability = 0.05
    num_parents = 10
    tolerance = 0.01

    # Create Genetic Algorithm instance
    ga = GeneticAlgorithm(
        populationSize=population_size,
        numBitsPerIndividual=num_bits_per_individual,
        numGenerations=num_generations,
        crossoverProbability=crossover_probability,
        mutationProbability=mutation_probability,
        numParents=num_parents,
        fitnessFunction=absolute_parameters_fitness,
        tolerance=tolerance,
    )

    decoded_params = np.zeros(28)

    # Create some dummy bikes (this step might be different depending on your actual Bike class implementation)
    for _ in range(population_size):
        bike = Bike(
            decoded_params[0],
            decoded_params[1],
            decoded_params[2],
            decoded_params[3],
            decoded_params[4],
            decoded_params[5],
            decoded_params[6],
            decoded_params[7],
            decoded_params[8],
            decoded_params[9],
            decoded_params[10],
            decoded_params[11],
            decoded_params[12],
            decoded_params[13],
            decoded_params[14],
            decoded_params[15],
            decoded_params[16],
            decoded_params[17],
            decoded_params[18],
            decoded_params[19],
            decoded_params[20],
            decoded_params[21],
            decoded_params[22],
            decoded_params[23],
            decoded_params[24],
            decoded_params[25],
            decoded_params[26],
            decoded_params[27],
        )
        ga.bikes.append(bike)
        ga.bike2array(len(ga.bikes) - 1, bike)

    # Run the genetic algorithm
    ga.fit()

    # Get and print statistics
    max_values, avg_values, min_values = ga.getStatistics()
    print("\nOptimization Results:")
    print("Max Fitness Values:", max_values)
    print("Average Fitness Values:", avg_values)
    print("Min Fitness Values:", min_values)

    # Get the best solution
    best_individual_index = np.argmax(ga.populationFitness)
    best_chromosome = ga.population[np.argmax(ga.populationFitness)]
    best_params = ga.binarytobike(best_chromosome)

    print("\nBest Bike Parameters:")
    param_names = [
        "wheel_1_x",
        "wheel_1_y",
        "wheel_1_radius",
        "wheel_1_mass",
        "wheel_1_torque",
        "wheel_2_x",
        "wheel_2_y",
        "wheel_2_radius",
        "wheel_2_mass",
        "wheel_2_torque",
        "body_1_x",
        "body_1_y",
        "body_1_mass",
        "body_2_x",
        "body_2_y",
        "body_2_mass",
        "k_spring_w1_w2",
        "k_spring_w1_b1",
        "k_spring_w1_b2",
        "k_spring_w2_b1",
        "k_spring_w2_b2",
        "k_spring_b1_b2",
        "loss_spring_w1_w2",
        "loss_spring_w1_b1",
        "loss_spring_w1_b2",
        "loss_spring_w2_b1",
        "loss_spring_w2_b2",
        "loss_spring_b1_b2",
    ]

    for name, value in zip(param_names, best_params):
        print(f"{name}: {value}")
    assert best_params[-1] < 0.01
