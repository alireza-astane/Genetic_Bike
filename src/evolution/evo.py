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
        chromozome = []
        chromozome.append(np.binary_repr(self.bikes[i].wheel_1_x, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_1_y, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_1_radius, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_1_mass, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_1_torque, self.numBitsPerIndividual))
        
        chromozome.append(np.binary_repr(self.bikes[i].wheel_2_x, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_2_y, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_2_radius, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_2_mass,self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].wheel_2_torque, self.numBitsPerIndividual))
        
        chromozome.append(np.binary_repr(self.bikes[i].body_1_x, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].body_1_y, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].body_1_mass, self.numBitsPerIndividual))
        
        chromozome.append(np.binary_repr(self.bikes[i].body_2_x, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].body_2_y, self.numBitsPerIndividual))
        chromozome.append(np.binary_repr(self.bikes[i].body_2_mass,self.numBitsPerIndividual))
        
       
        self.population[i] = chromozome
        
    
        
  
    def fit(self):
        self.calculateFitness()
        for i in range(self.numGenerations):
            
            self.elites[i] = self.population[np.argmax(self.populationFitness)].copy()
            self.selection()
            self.crossover()
            self.mutation()
            # Update fitness
            self.population[np.argmin(self.populationFitness)] = self.elites[i].copy()

            self.calculateStatistics()
            self.calculateFitness()

            if self.tolerance < self.maxValues[-1]:
                print("Last iteration", i)
                self.elites[i] = self.population[np.argmax(self.populationFitness)]
                break

    def calculateFitness(self):
        for i in range(self.populationSize):
            self.populationFitness[i] = self.fitnessFunction(self, self.population[i])

    def binaryToDecimal(self, individual, lowerBound, upperBound):
        x = 0
        for k, bit in enumerate(individual[::-1]):
            x += (bit * 2**k)

        x = lowerBound + ((upperBound - lowerBound) / (2**self.numBitsPerIndividual - 1)) * x
        return x
        
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
