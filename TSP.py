# use hamiltonian cycle to solve TSP problem

import random

import matplotlib.pyplot as plt
import numpy as np


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
    child = Tour(child)
    return child


def swap_mutation(tour):
    index1 = random.randint(0, len(tour.city_list) - 1)
    index2 = random.randint(0, len(tour.city_list) - 1)
    while index1 == index2:
        index2 = random.randint(0, len(tour.city_list) - 1)
    tour.city_list[index1], tour.city_list[index2] = tour.city_list[index2], tour.city_list[index1]
    return tour


class GA:
    def __init__(self, city_list, population_size, elite_size, mutation_rate, generations):
        self.city_list = city_list
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initial_population()
        self.best_tour = self.population.tour_list[0]
        self.best_distance = self.best_tour.distance
        self.best_generation = 0
        self.distance_list = []

    def initial_population(self):
        tour_list = []
        for i in range(self.population_size):
            tour = Tour(random.sample(self.city_list, len(self.city_list)))
            tour_list.append(tour)
        population = Population(tour_list)
        return population

    def selection(self):
        fitness = self.population.fitness
        fitness_sum = sum(fitness)
        fitness = [i / fitness_sum for i in fitness]
        fitness_cumsum = np.cumsum(fitness)
        fitness_cumsum[-1] = 1
        fitness_cumsum = list(fitness_cumsum)
        new_tour_list = []
        for i in range(self.elite_size):
            new_tour_list.append(self.population.tour_list[i])
        for i in range(self.population_size - self.elite_size):
            random_num = random.random()
            for j in range(self.population_size):
                if random_num <= fitness_cumsum[j]:
                    new_tour_list.append(self.population.tour_list[j])
                    break
        self.population.tour_list = new_tour_list

    def crossover(self):
        new_tour_list = []
        for i in range(self.elite_size):
            new_tour_list.append(self.population.tour_list[i])
        for i in range(self.population_size - self.elite_size):
            parent1 = random.choice(self.population.tour_list)
            parent2 = random.choice(self.population.tour_list)
            child = order_crossover(parent1, parent2)
            new_tour_list.append(child)
        self.population.tour_list = new_tour_list

    def mutation(self):
        for i in range(self.elite_size, self.population_size):
            if random.random() < self.mutation_rate:
                self.population.tour_list[i] = swap_mutation(self.population.tour_list[i])

    def evolve(self):
        for i in range(self.generations):
            self.selection()
            self.crossover()
            self.mutation()
            self.population = Population(self.population.tour_list)
            self.best_tour = self.population.tour_list[0]
            self.best_distance = self.best_tour.distance
            self.best_generation = i
            self.distance_list.append(self.best_distance)

    def plot(self):
        plt.plot(self.distance_list)
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.show()

    def __repr__(self):
        return str(self.population)


# define the function of main and visualize the hamiltonian circuit in 3d space
def main():
    city_list = []
    for i in range(20):
        city = City(x=random.randint(0, 100), y=random.randint(0, 100))
        city_list.append(city)
    ga = GA(city_list, 100, 10, 0.01, 100)
    ga.evolve()
    ga.plot()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(ga.best_tour.city_list)):
        ax.scatter(ga.best_tour.city_list[i].x, ga.best_tour.city_list[i].y, c='r')
    for i in range(len(ga.best_tour.city_list)):
        ax.plot([ga.best_tour.city_list[i].x, ga.best_tour.city_list[(i + 1) % len(ga.best_tour.city_list)].x],
                [ga.best_tour.city_list[i].y, ga.best_tour.city_list[(i + 1) % len(ga.best_tour.city_list)].y], c='r')
    plt.show()


if __name__ == "__main__":
    main()
