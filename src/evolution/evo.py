import numpy as np
import random


class GeneticAlgorithm():
    def __init__(self, populationSize, numBitsPerIndividual, numGenerations, crossoverProbability, mutationProbability, numParents, fitnessFunction, tolerance, numCompetitors=2):

        self.numBitsPerIndividual = numBitsPerIndividual
        self.numParents = numParents
        self.populationSize = populationSize
        self.crossoverProbability = crossoverProbability
        self.mutationProbability = mutationProbability
        self.numGenerations = numGenerations
        self.numCompetitors = numCompetitors
        
        self.population = np.random.randint(0, 2, (self.populationSize, self.numBitsPerIndividual))
        self.populationFitness = np.zeros(self.populationSize)
        self.parents = np.zeros((self.numParents, self.numBitsPerIndividual))
        self.bikes = []
        
        self.fitnessFunction = fitnessFunction

        self.elites = np.zeros((self.numGenerations, self.numBitsPerIndividual))
        self.maxValues = []
        self.averageValues = []
        self.minValues = []

        self.tolerance = tolerance
        

    def bike2array(self, i, bike):
        chrom = []
        scale = 10  
        nbit=self.numBitsPerIndividual
        
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_1_x * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_1_y * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_1_radius * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_1_mass * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_1_torque * scale), nbit))
        
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_2_x * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_2_y * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_2_radius * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_2_mass * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].wheel_2_torque * scale), nbit))
        
        chrom.append(np.binary_repr(int(self.bikes[i].body_1_x * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].body_1_y * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].body_1_mass * scale), nbit))
        
        chrom.append(np.binary_repr(int(self.bikes[i].body_2_x * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].body_2_y * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].body_2_mass * scale), nbit))

        chrom.append(np.binary_repr(int(self.bikes[i].k_spring_w1_w2 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].k_spring_w1_b1 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].k_spring_w1_b2 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].k_spring_w2_b1 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].k_spring_w2_b2 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].k_spring_b1_b2 * scale), nbit))

        chrom.append(np.binary_repr(int(self.bikes[i].loss_spring_w1_w2 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].loss_spring_w1_b1 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].loss_spring_w1_b2 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].loss_spring_w2_b1 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].loss_spring_w2_b2 * scale), nbit))
        chrom.append(np.binary_repr(int(self.bikes[i].loss_spring_b1_b2 * scale), nbit))

        self.population[i] = chrom

    def binarytobike(self, chrom):
        scale = 10
        deci = []
        for binary in chrom:
            decimal = int(str(binary), 2)
            deci.append(decimal / scale)
        return deci
 
    def fit(self):

        print("Initial population:")
        print(self.population)
        
        self.calculateFitness()
        print("Initial fitness:")
        print(self.populationFitness)

        for i in range(self.numGenerations):
            print(f"\nGeneration {i + 1}:")
            
            self.elites[i] = self.population[np.argmax(self.populationFitness)].copy()
            print("Elites:")
            print(self.elites[i])

            self.selection()
            print("Selected Parents:")
            print(self.parents)

            self.crossover()
            print("Population after crossing:")
            print(self.population)

            self.mutation()
            print("Population after mutation:")
            print(self.population)
            # Update fitness
            self.population[np.argmin(self.populationFitness)] = self.elites[i].copy()

            self.calculateStatistics()
            self.calculateFitness()
            print("Generation Fitness:")
            print(self.populationFitness)

            if self.tolerance < self.maxValues[-1]:
                print("Last iteration", i)
                self.elites[i] = self.population[np.argmax(self.populationFitness)]
                break

    def calculateFitness(self):
        for i in range(self.populationSize):
            self.populationFitness[i] = self.fitnessFunction(self, self.population[i])

    def selection(self):
        # Tournament
        for i in range(self.numParents):
            competitorIndices = np.random.randint(0, len(self.populationFitness), (self.numCompetitors, 1))
            winnerIndex = np.argmax(self.populationFitness[competitorIndices])
            self.parents[i] = self.population[competitorIndices[winnerIndex]].copy()

    def crossover(self):
        for i in range(self.populationSize // 2):
            parent1 = self.parents[random.randint(0, len(self.parents)) - 1].copy()
            parent2 = self.parents[random.randint(0, len(self.parents)) - 1].copy()
                        
            if random.random() < self.crossoverProbability:
                offspring = self._cross(parent1, parent2)
            else:
                offspring = (parent1, parent2)
            
            self.population[2 * i] = offspring[0]
            self.population[2 * i + 1] = offspring[1]
    
    def _cross(self, parent1, parent2):
        crossoverPoint = random.randint(0, self.numBitsPerIndividual)
        offspring1 = np.concatenate((parent1[:crossoverPoint], parent2[crossoverPoint:]), axis=None)
        offspring2 = np.concatenate((parent2[:crossoverPoint], parent1[crossoverPoint:]), axis=None)
        return (offspring1, offspring2)

    def mutation(self):
        # Possible vectorized implementation
        for i in range(self.populationSize):
            for j in range(self.numBitsPerIndividual):
                if random.random() < self.mutationProbability:
                    self.population[i][j] = self.population[i][j] ^ 1  # XOR operator: 1^1 = 0, 0^1 = 1
    
    def getElites(self):
        return self.elites

    def calculateStatistics(self):
        self.maxValues.append(max(self.populationFitness))
        print(self.maxValues[-1])
        self.averageValues.append(np.average(self.populationFitness))
        self.minValues.append(min(self.populationFitness))

    def getStatistics(self):
        return self.maxValues, self.averageValues, self.minValues
