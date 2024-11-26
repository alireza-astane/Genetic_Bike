import numpy as np
import random


class AlgoritmoGenetico():
    def __init__(self,tamañoPoblacion, numBitsIndividuo, numGeneraciones, probabilidadCruce, probabilidadMutacion, numPadres, funcionFitness,tol,numCompetidores=2):

        self.numBitsIndividuo = numBitsIndividuo
        self.numPadres = numPadres
        self.tamañoPoblacion = tamañoPoblacion
        self.probablidadCruce = probabilidadCruce
        self.probabilidadMutacion = probabilidadMutacion
        self.numGeneraciones=numGeneraciones
        self.numCompetidores = numCompetidores
        
        self.poblacion = np.random.randint(0,2,(self.tamañoPoblacion,self.numBitsIndividuo))
        self.poblacionFitness = np.zeros(self.tamañoPoblacion)
        self.padres = np.zeros((self.numPadres,self.numBitsIndividuo))

        self.funcionFitness = funcionFitness

        self.elites = np.zeros((self.numGeneraciones,self.numBitsIndividuo))
        self.maximos = []
        self.promedio = []
        self.minimo = []

        self.tol = tol
        
    def fit(self):
        self.calcularFitness()
        for i in range(self.numGeneraciones):
            
            self.elites[i] = self.poblacion[np.argmax(self.poblacionFitness)].copy()
            self.seleccion()
            self.cruze()
            self.mutacion()
            # ACA CALCULAR FITNESS
            self.poblacion[np.argmin(self.poblacionFitness)] = self.elites[i].copy()

            self.calcularEstadistica()
            self.calcularFitness()

            if self.tol < self.maximos[-1]:
                print("Ultima iteración",i)
                self.elites[i] = self.poblacion[np.argmax(self.poblacionFitness)]
                break

    def calcularFitness(self):
        for i in range(self.tamañoPoblacion):
            self.poblacionFitness[i] = self.funcionFitness(self,self.poblacion[i])


    def binarioADecimal(self,ind,menor,mayor):
        x = 0
        for k,i in enumerate(ind[::-1]):
            x += (i * 2**k)

        x = menor + ((mayor-menor)/ (2**self.numBitsIndividuo-1)) * x
        return x
        
    def seleccion(self):
        #Torneo
        for i in range(self.numPadres):
            competidores_index = np.random.randint(0,len(self.poblacionFitness),(self.numCompetidores,1))
            ganador_index = np.argmax(self.poblacionFitness[competidores_index])
            self.padres[i] = self.poblacion[competidores_index[ganador_index]].copy()

    def cruze(self):
        for i in range(self.tamañoPoblacion//2):
            padre1 = self.padres[random.randint(0,len(self.padres))-1].copy()
            padre2 = self.padres[random.randint(0,len(self.padres))-1].copy()
                        
            if random.random() < self.probablidadCruce:
                hijos = self._cruzar(padre1,padre2)
            else:
                hijos = (padre1, padre2)
            
            self.poblacion[2*i] = hijos[0]
            self.poblacion[2*i+1] = hijos[1]
    
    def _cruzar(self,padre1,padre2):
        puntoCruze = random.randint(0,self.numBitsIndividuo)
        hijo1 = np.concatenate((padre1[:puntoCruze], padre2[puntoCruze:]),axis=None)
        hijo2 = np.concatenate((padre2[:puntoCruze] , padre1[puntoCruze:]),axis=None)
        return (hijo1, hijo2)

    def mutacion(self):
        # Posibilidad de hacerlo vectorial
        for i in range(self.tamañoPoblacion):
            for j in range(self.numBitsIndividuo):
                if random.random() < self.probabilidadMutacion:
                   self.poblacion[i][j] = self.poblacion[i][j] ^ 1 #Operador XOR: 1^1 = 0, 0^1 = 1
    
    def obtenerElites(self):
        return self.elites

    def calcularEstadistica(self):
        self.maximos.append(max(self.poblacionFitness))
        print(self.maximos[-1])
        self.promedio.append(np.average(self.poblacionFitness))
        self.minimo.append(min(self.poblacionFitness))

    def obtenerEstadisticas(self):
        return self.maximos, self.promedio, self.minimo