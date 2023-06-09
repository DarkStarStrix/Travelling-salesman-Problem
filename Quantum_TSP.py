import random

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer


# define the class of city
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# define the class of tour
class Tour:
    def __init__(self, city_list):
        self.city_list = city_list
        self.distance = self.get_distance()

    def get_distance(self):
        distance = 0
        for i in range(len(self.city_list)):
            distance += self.city_list[i].distance(self.city_list[(i + 1) % len(self.city_list)])
        return distance

    def __repr__(self):
        return str(self.city_list) + " " + str(self.distance)


# define the class of population
class Population:
    def __init__(self, tour_list):
        self.tour_list = tour_list
        self.fitness = self.get_fitness()

    def get_fitness(self):
        fitness = []
        for i in range(len(self.tour_list)):
            fitness.append(1 / self.tour_list[i].distance)
        return fitness

    def __repr__(self):
        return str(self.tour_list) + " " + str(self.fitness)


# define the class of GA
def order_crossover(parent1, parent2):
    child = [None] * len(parent1.city_list)
    start_index = random.randint(0, len(parent1.city_list) - 1)
    end_index = random.randint(start_index, len(parent1.city_list) - 1)
    for i in range(start_index, end_index + 1):
        child[i] = parent1.city_list[i]
    for i in range(len(parent2.city_list)):
        if parent2.city_list[i] not in child:
            for j in range(len(child)):
                if child[j] is None:
                    child[j] = parent2.city_list[i]
                    break
    return Tour(child)


def mutation(tour):
    index1 = random.randint(0, len(tour.city_list) - 1)
    index2 = random.randint(0, len(tour.city_list) - 1)
    tour.city_list[index1], tour.city_list[index2] = tour.city_list[index2], tour.city_list[index1]
    return tour


def selection(population):
    fitness_sum = sum(population.fitness)
    probability = [fitness / fitness_sum for fitness in population.fitness]
    index = random.choices(range(len(population.tour_list)), probability)[0]
    return population.tour_list[index]


def GA(population, generation):
    for i in range(generation):
        new_tour_list = []
        for j in range(len(population.tour_list)):
            parent1 = selection(population)
            parent2 = selection(population)
            child = order_crossover(parent1, parent2)
            child = mutation(child)
            new_tour_list.append(child)
        population = Population(new_tour_list)
    return population


# define the class of QGA
def create_circuit(city_list):
    q = QuantumRegister(len(city_list))
    c = ClassicalRegister(len(city_list))
    qc = QuantumCircuit(q, c)
    qc.h(q)
    qc.barrier()
    qc.measure(q, c)
    return qc


def get_fitness(counts, city_list):
    fitness = []
    for state in counts:
        city_index = [int(i) for i in state]
        city_list = [city_list[i] for i in city_index]
        tour = Tour(city_list)
        np.seterr(divide='ignore', invalid='ignore')
        fitness.append(1 / tour.distance)
    return fitness


def QGA(city_list, generation):
    qc = create_circuit(city_list)
    backend = Aer.get_backend('qasm_simulator')
    shots = 1000
    for i in range(generation):
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        fitness = get_fitness(counts, city_list)
        index = fitness.index(max(fitness))
        city_index = [int(i) for i in list(counts.keys())[index]]
        city_list = [city_list[i] for i in city_index]
    return city_list


# plot the result of GA and QGA on the same figure on the graph draw the line between the cities in 3d space
def plot(city_list, algorithm):
    global city
    x = []
    y = []
    for city in city_list:
        x.append(city.x)
        y.append(city.y)
    x.append(city_list[0].x)
    y.append(city_list[0].y)
    plt.plot(x, y, 'ro-')
    plt.title(algorithm)
    plt.show()


# define the main function
def main():
    city_list = []
    for i in range(10):
        city_list.append(City(random.randint(0, 100), random.randint(0, 100)))
    print("The city list is: ", city_list)
    tour_list = []
    for i in range(100):
        tour = Tour(random.sample(city_list, len(city_list)))
        tour_list.append(tour)
    population = Population(tour_list)
    population = GA(population, 100)
    print("The result of GA is: ", population.tour_list[0])
    plot(population.tour_list[0].city_list, "GA")
    city_list = QGA(city_list, 100)
    print("The result of QGA is: ", city_list)
    plot(city_list, "QGA")

    # run the main function


if __name__ == "__main__":
    main()
