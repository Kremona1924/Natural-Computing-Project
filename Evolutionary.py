import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import NN
import copy
import math

from simulation import boids_sim

class EA:
    def __init__(self, pop_size, layer_sizes) -> None:
        self.pop = self.initialise_population(pop_size, layer_sizes)

    def initialise_population(self, pop_size, layer_sizes):
        pop = []
        for id in range(pop_size):
            pop.append({"params" : NN.initialise_network(layer_sizes),
                "fitness" : 0,
                "original_id": id,
                "id" : id})
        return pop


    def run(self, num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log = False, plot_chart=False, save_population = False, show_screen = False):
        for i in range(num_generations):
            sim = boids_sim(self.pop)
            alignments, cohesions = sim.run_with_screen(num_steps, plot_chart = plot_chart, show_screen=show_screen, log=log, filename="simulation_log.json")
            self.compute_fitness(alignments, cohesions)

            if log:
                print("Generation: ", i)
                print("Fittest individual: ", min(self.pop, key=lambda x: x["fitness"])["fitness"])
            self.pop = self.create_new_population(mutation_rate, mutation_step, tournament_size)
        
        # Run and evaluate one last time
        alignments, cohesions = sim.run_with_screen(num_steps, plot_chart = plot_chart, show_screen=show_screen, log=log, filename="simulation_log.json")
        self.compute_fitness(alignments, cohesions)
        
        if save_population:
            self.save_population()

    def compute_fitness(self, alignments, cohesions): #TODO finetune fitness function
        for i, agent in enumerate(self.pop):
            alignment = np.mean(alignments[i][-100:]) # Mean of last 30 alignment values for this agent
            cohesion = np.mean(cohesions[i][-100:]) # Mean of last 30 cohesion values for this agent
            agent["fitness"] = cohesion * 20 + alignment

    # def create_new_population(self, mu, ms, k):
    #     new_pop = []
    #     for id in range(len(self.pop)):
    #         parent = self.tournament_selection(self.pop, k, 1)
    #         offspring = copy.deepcopy(parent)
    #         self.mutate(offspring["params"], mu, ms)
    #         offspring["id"] = id
    #         # TODO: Add crossover
    #         new_pop.append(offspring)
    #     return new_pop

    def create_new_population(self, mu, ms, k):
        new_pop = []
        while len(new_pop) < len(self.pop):
            # Select two parents via tournament selection
            parent1 = self.tournament_selection(self.pop, k, 1)
            parent2 = self.tournament_selection(self.pop, k, 1)

            # Generates child via crossover
            child = self.crossover(parent1, parent2)

            # Mutation on child
            self.mutate(child['params'], mu, ms)

            # Adds child to new population
            child['id'] = len(new_pop)
            new_pop.append(child)
            if len(new_pop) < len(self.pop):  # If extra childs needed, reverse parents (no difference)
                child = self.crossover(parent2, parent1)
                self.mutate(child['params'], mu, ms)
                child['id'] = len(new_pop)
                new_pop.append(child)
        return new_pop
    
    def tournament_selection(self, pop, tournament_size, num):
        winners = []
        for _ in range(num):
            selection = np.random.choice(len(pop), size=tournament_size, replace=False) # Indices of individiuals in the tournament
            winner = selection[np.argmin([pop[i]["fitness"] for i in selection])] # Find index of fittest individual
            winners.append(pop[winner]) # Add fittest individual to output
        return winners[0] # TODO: fix output type to not be list when num = 1
    
    def mutate(self, params, mutation_rate, mutation_step):
        w, b = params
        for weight in w:
            if np.random.random() < mutation_rate:
                weight += np.random.normal(0, mutation_step) # TODO: set mutation size

        for bias in b:
            if np.random.random() < mutation_rate:
                bias += np.random.normal(0, mutation_step) # TODO: set mutation size
        return w, b
    
    def crossover(self, parent1, parent2):
        child_weights = []
        child_biases = []


        for w1, w2, b1, b2 in zip(parent1['params'][0], parent2['params'][0], parent1['params'][1], parent2['params'][1]):
            # creates a matrix with false and true values depending on > or < 0.5
            mask = np.random.rand(*w1.shape) > 0.5
            child_weight = np.where(mask, w1, w2)
            child_weights.append(child_weight)
            
            # creates an array with false and true values depending on > or < 0.5
            mask = np.random.rand(b1.size) > 0.5
            child_bias = np.where(mask, b1, b2)
            child_biases.append(child_bias)
        
        return {'params': (child_weights, child_biases)}

    def save_fitness(self, pop):
        # Save fitness values of the population. At each step, all fitness values are printed to one line
        pop_fitness = [i["fitness"] for i in pop]
        with open("fitness_over_time.txt", "a") as f:
            np.savetxt(f,pop_fitness, newline=' ')
            f.write("\n")
    
    def save_population(self):
        weights = np.array([i["params"][0] for i in self.pop], dtype=object)
        biases = np.array([i["params"][1] for i in self.pop], dtype=object)
        fitnesses = np.array([i["fitness"] for i in self.pop], dtype=object)
        np.savez("population_info", weights = weights, biases = biases, fitnesses = fitnesses)