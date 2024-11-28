from src.evolution.evo import GeneticAlgorithm
from  src.bike.bike import Bike
import numpy as np
import random
import matplotlib as plt

def absolute_parameters_fitness(ga_instance, chromosome):
    """
    Fitness function to minimize the sum of absolute values of bike parameters.
    The goal is to get as close to 0 for all parameters as possible.
    """
    # Convert binary chromosome to decimal values
    decoded_params = ga_instance.binarytobike(chromosome)
    
    # Create a Bike instance with the decoded parameters
    bike = Bike(
        wheel_1_x=decoded_params[0], wheel_1_y=decoded_params[1], 
        wheel_1_radius=decoded_params[2], wheel_1_mass=decoded_params[3], 
        wheel_1_torque=decoded_params[4],
        wheel_2_x=decoded_params[5], wheel_2_y=decoded_params[6], 
        wheel_2_radius=decoded_params[7], wheel_2_mass=decoded_params[8], 
        wheel_2_torque=decoded_params[9],
        body_1_x=decoded_params[10], body_1_y=decoded_params[11], 
        body_1_mass=decoded_params[12],
        body_2_x=decoded_params[13], body_2_y=decoded_params[14], 
        body_2_mass=decoded_params[15]
    )
    
    # Calculate sum of absolute values
    total_abs_sum = (
        abs(bike.wheel_1_x) + abs(bike.wheel_1_y) + 
        abs(bike.wheel_1_radius) + abs(bike.wheel_1_mass) + 
        abs(bike.wheel_1_torque) +
        abs(bike.wheel_2_x) + abs(bike.wheel_2_y) + 
        abs(bike.wheel_2_radius) + abs(bike.wheel_2_mass) + 
        abs(bike.wheel_2_torque) +
        abs(bike.body_1_x) + abs(bike.body_1_y) + abs(bike.body_1_mass) +
        abs(bike.body_2_x) + abs(bike.body_2_y) + abs(bike.body_2_mass)
    )
    
    # The fitness is the negative of the sum (so minimizing absolute values maximizes fitness)
    return -total_abs_sum

def main():
    # Genetic Algorithm parameters
    population_size = 50
    num_bits_per_individual = 16  # Adjust based on your scale and precision needs
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
        tolerance=tolerance
    )

    # Create some dummy bikes (this step might be different depending on your actual Bike class implementation)
    for _ in range(population_size):
        bike = Bike()
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
        "wheel_1_x", "wheel_1_y", "wheel_1_radius", "wheel_1_mass", "wheel_1_torque",
        "wheel_2_x", "wheel_2_y", "wheel_2_radius", "wheel_2_mass", "wheel_2_torque",
        "body_1_x", "body_1_y", "body_1_mass",
        "body_2_x", "body_2_y", "body_2_mass"
    ]
    
    for name, value in zip(param_names, best_params):
        print(f"{name}: {value}")

    # Optional: Plot fitness progression
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_values, label='Max Fitness')
    plt.plot(avg_values, label='Average Fitness')
    plt.plot(min_values, label='Min Fitness')
    plt.title('Fitness Progression')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()