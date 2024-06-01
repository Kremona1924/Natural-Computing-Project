import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import NN
import copy
import math

from simulation import boids_sim

class EA:
    def __init__(self, pop_size, layer_sizes, crossover_type) -> None:
        self.pop = self.initialise_population(pop_size, layer_sizes)
        self.crossover_type = crossover_type
    
    def crossover(self, parent1, parent2):
        if self.crossover_type == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_type == 'single_point':
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_type == 'mean':
            return self.mean_crossover(parent1, parent2)
        elif self.crossover_type == 'none':
            return self.no_crossover(parent1, parent2)

    def initialise_population(self, pop_size, layer_sizes):
        pop = []
        for id in range(pop_size):
            pop.append({"params" : NN.initialise_network(layer_sizes),
                "original_id": id,
                "id" : id,
                "fitness" : 0,
                "cohesion_score": 0,
                "alignment_score": 0
            })
        return pop


    def run(self, num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log = False, plot_chart=False, save_population = False, show_screen = False, save_file_extension = ""):
        for i in range(num_generations):
            sim = boids_sim(self.pop)
            alignments, cohesions = sim.run_with_screen(num_steps, plot_chart = plot_chart, show_screen=show_screen, log=log, filename="simulation_log.json")
            self.compute_fitness(alignments, cohesions)

            if log:
                print("Generation: ", i)
                print("Cohesion: ", np.mean([a["cohesion_score"] for a in self.pop]))
                print("Alignment: ", np.mean([a["alignment_score"] for a in self.pop]))
                print("Fittest individual: ", min(self.pop, key=lambda x: x["fitness"])["fitness"])
                print("\n")
                self.save_fitness(save_file_extension)

            self.pop = self.create_new_population(mutation_rate, mutation_step, tournament_size)
        
        # Run and evaluate one last time
        alignments, cohesions = sim.run_with_screen(num_steps, plot_chart = plot_chart, show_screen=show_screen, log=log, filename="simulation_log.json")
        self.compute_fitness(alignments, cohesions)
        
        if save_population:
            self.save_population(save_file_extension)

    def compute_fitness(self, alignments, cohesions): #TODO finetune fitness function
        for i, agent in enumerate(self.pop):
            alignment = np.mean(alignments[i][-100:]) # Mean of last 30 alignment values for this agent
            cohesion = np.mean(cohesions[i][-100:]) # Mean of last 30 cohesion values for this agent
            cohesion_score = (cohesion**2) * 10
            
            agent["cohesion_score"] = cohesion_score
            agent["alignment_score"] = alignment
            agent["fitness"] = cohesion_score + alignment

    def create_new_population(self, mu, ms, k):
        new_pop = []
        for id in range(len(self.pop)):
            # no crossover population
            if self.crossover_type == 'none':
                parent = self.tournament_selection(self.pop, k, 1)
                offspring = copy.deepcopy(parent)
                self.mutate(offspring["params"], mu, ms)
                offspring["id"] = id
            # crossover population
            else:
                parent1 = self.tournament_selection(self.pop, k, 1)
                parent2 = self.tournament_selection(self.pop, k, 1)
                # opnieuw tot twee verschillende ouders voor crossover
                while parent1['id'] == parent2['id']:
                    parent2 = self.tournament_selection(self.pop, k, 1)
                offspring = self.crossover(parent1, parent2)
                self.mutate(offspring["params"], mu, ms)
                offspring["id"] = id
            new_pop.append(offspring)
        return new_pop

    
    def tournament_selection(self, pop, tournament_size, num):
        winners = []
        for _ in range(num):
            selection = np.random.choice(len(pop), size=tournament_size, replace=False) # Indices of individiuals in the tournament
            winner = selection[np.argmin([pop[i]["fitness"] for i in selection])] # Find index of fittest individual
            winners.append(pop[winner]) # Add fittest individual to output
        return winners[0] # TODO: fix output type to not be list when num = 1
    
    def fitness_proportionate_selection(self, num):
        fitnesses = np.array([i["fitness"] for i in self.pop])
        scaled = -fitnesses + np.min(fitnesses) + np.max(fitnesses)
        probabilities = scaled / np.sum(scaled)
        return np.random.choice(self.pop, num, p=probabilities)[0]

    
    def mutate(self, params, mutation_rate, mutation_step):
        w, b = params
        for weight in w:
            if np.random.random() < mutation_rate:
                weight += np.random.normal(0, mutation_step) # TODO: set mutation size

        for bias in b:
            if np.random.random() < mutation_rate:
                bias += np.random.normal(0, mutation_step) # TODO: set mutation size
        return w, b
    
    def no_crossover(self, parent1, parent2):
        # random keuze tussen parent1 of parent2
        chosen_parent = parent1 if np.random.rand() > 0.5 else parent2
        return {'params': copy.deepcopy(chosen_parent['params'])}

    def uniform_crossover(self, parent1, parent2):
        child_weights = []
        child_biases = []
        for w1, w2, b1, b2 in zip(parent1['params'][0], parent2['params'][0], parent1['params'][1], parent2['params'][1]):
            # mask matrix met true/false values voor parent 1 of 2
            mask = np.random.rand(*w1.shape) > 0.5
            child_weight = np.where(mask, w1, w2)
            child_weights.append(child_weight)
            
            # mask matrix met true/false values voor parent 1 of 2
            mask = np.random.rand(b1.size) > 0.5 
            child_bias = np.where(mask, b1, b2)
            child_biases.append(child_bias)
        
        return {'params': (child_weights, child_biases)}

    def single_point_crossover(self, parent1, parent2):
        child_weights = []
        child_biases = []
        
        for w1, w2, b1, b2 in zip(parent1['params'][0], parent2['params'][0], parent1['params'][1], parent2['params'][1]):
            # Random crossover point
            crossover_point_w = np.random.randint(0, w1.shape[0])
            crossover_point_b = np.random.randint(0, b1.shape[0])

            child_weights.append(np.concatenate((w1[:crossover_point_w], w2[crossover_point_w:])))
            child_biases.append(np.concatenate((b1[:crossover_point_b], b2[crossover_point_b:])))
        
        return {'params': (child_weights, child_biases)}

    def mean_crossover(self, parent1, parent2):

        w = [(w1 + w2)/2 for w1, w2 in zip(parent1["params"][0], parent2["params"][0])]
        b = [(b1 + b2)/2 for b1, b2 in zip(parent1["params"][1], parent2["params"][1])]

        return {'params': (w, b)}

    def save_fitness(self, save_file_extension):
        # Save fitness values of the population. At each step, all fitness values are printed to one line
        pop_fitness = [i["fitness"] for i in self.pop]
        with open("fitness_over_time" + save_file_extension + ".txt", "a") as f:
            np.savetxt(f,pop_fitness, newline=' ')
            f.write("\n")
    
    def save_population(self, save_file_extension):
        weights = np.array([i["params"][0] for i in self.pop], dtype=object)
        biases = np.array([i["params"][1] for i in self.pop], dtype=object)
        fitnesses = np.array([i["fitness"] for i in self.pop], dtype=object)
        np.savez("population_info" + save_file_extension, weights = weights, biases = biases, fitnesses = fitnesses)