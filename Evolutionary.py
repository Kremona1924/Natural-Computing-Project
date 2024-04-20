import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import NN
import copy

global id_counter
id_counter = 0

class EA:
    def __init__(self, pop_size, layer_sizes) -> None:
        self.pop = self.initialise_population(pop_size, layer_sizes)

    def run(self, num_generations, mutation_rate, tournament_size, log = False):
        for i in range(num_generations):
            self.evalutate_pop()
            if log:
                print("Generation: ", i)
                print("Fittest individual: ", max(self.pop, key=lambda x: x["fitness"])["fitness"])
            self.pop = self.create_new_population(mutation_rate, tournament_size)

    def crossover(self, x, y):
        return # TODO: Implement crossover

    def mutate(self, params, mutation_rate):
        w, b = params
        for weight in w:
            if np.random.random() < mutation_rate:
                weight += np.random.normal(0, 0.05) # TODO: set mutation size

        for bias in b:
            if np.random.random() < mutation_rate:
                bias += np.random.normal(0, 0.05) # TODO: set mutation size
        return w, b

    def fitness(self, x): 
        # Test fitness function- adding both inputs
        fitness = 0
        for i in range(10):
            input = np.random.rand(1, 2) - 0.5
            output = NN.feed_forward(x, input)
            fitness -= (np.sum(input) - output)**2
        return fitness

    def evalutate_pop(self):
        for network in self.pop:
            network["fitness"] = self.fitness(network["params"])

    def initialise_population(self, pop_size, layer_sizes):
        global id_counter
        pop = []
        for _ in range(pop_size):
            pop.append({"params" : NN.initialise_network(layer_sizes),
                "fitness" : 0,
                "id" : id_counter})
            id_counter += 1
        return pop
        
    def tournament_selection(self, pop, tournament_size, num):
        winners = []
        for _ in range(num):
            selection = np.random.choice(len(pop), size=tournament_size, replace=False) # Indices of individiuals in the tournament
            winner = selection[np.argmax([pop[i]["fitness"] for i in selection])] # Find index of fittest individual
            winners.append(pop[winner]) # Add fittest individual to output
        return winners[0] # TODO: fix output type to not be list when num = 1

    def create_new_population(self, mu, k):
        new_pop = []
        for i in range(len(self.pop)):
            parent = self.tournament_selection(self.pop, k, 1)
            offspring = copy.deepcopy(parent)
            self.mutate(offspring["params"], mu)
            # TODO: Add crossover
            new_pop.append(offspring)
        return new_pop

    def save_fitness(self, pop):
            # Save fitness values of the population. At each step, all fitness values are printed to one line
            pop_fitness = [i["fitness"] for i in pop]
            with open("fitness_over_time.txt", "a") as f:
                np.savetxt(f,pop_fitness, newline=' ')
                f.write("\n")
