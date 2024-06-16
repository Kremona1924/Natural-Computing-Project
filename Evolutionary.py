import os
import copy
import json
import numpy as np

import NN
from simulation import BoidsSim

'''
This file contains the Evolutionary Algorithm class called EA and the logging class called LogPerformance. The EA class runs the 
whole process of the EA and logs the output. It calls the BoidsSim class from simulation.py to run the simulation.
'''

def create_log_dir(dir="logs/experiment", suffix_len=2):
    # Check if run dir already exists, add 1 until it doesn't and make new dir.
    suffix=1
    log_dir = f"{dir}{str(suffix).zfill(suffix_len)}"
    while os.path.isdir(log_dir):
        suffix += 1
        log_dir = f"{dir}{str(suffix).zfill(suffix_len)}"
    
    os.mkdir(log_dir)
    return log_dir

class LogPerformance:
    def __init__(self, num_gens, num_steps, mr, ms, ts, ct):
        self.performance = {
                "number_of_generations": num_gens,
                "number_of_steps": num_steps,
                "mutation_rate": mr,
                "mutation_step": ms,
                "tournament_size": ts,
                "crossover_type": ct,
                "mean_cohesion_over_time": [],
                "mean_alignment_over_time": [],
                "mean_fitness_over_time": [],
                "fittest_individual_over_time": []
            }
    
    def update(self, pop):
        mean_cohesion = np.mean([a["cohesion_score"] for a in pop])
        mean_alignment = np.mean([a["alignment_score"] for a in pop])
        mean_fitness = np.mean([a["fitness"] for a in pop])
        fittest_individual = min(pop, key=lambda x: x["fitness"])["fitness"]

        self.performance["mean_cohesion_over_time"].append(mean_cohesion)
        self.performance["mean_alignment_over_time"].append(mean_alignment)
        self.performance["mean_fitness_over_time"].append(mean_fitness)
        self.performance["fittest_individual_over_time"].append(fittest_individual)

        return mean_cohesion, mean_alignment, fittest_individual

    def get_performance(self):
        return self.performance


class EA:
    def __init__(self, pop_size, layer_sizes, crossover_type, log_dir) -> None:
        self.pop = self.initialise_population(pop_size, layer_sizes)
        self.crossover_type = crossover_type

        self.log_dir = log_dir
        
    
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

    def run(self, num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log_states = 'log_none', log_perf=False, show_sim = False, save_population = False, save_file_extension = "nospec"):
    
        track_performance = LogPerformance(num_generations, num_steps, mutation_rate, mutation_step, tournament_size, self.crossover_type)
        
        for i in range(num_generations):
            sim = BoidsSim(self.pop)
            alignments, cohesions = sim.run_simulation(num_steps, show_sim=show_sim, log_states=log_states, filename=save_file_extension, log_dir=self.log_dir, last_gen=False)
            self.compute_fitness(alignments, cohesions)

            mc, ma, fi = track_performance.update(self.pop)

            if show_sim:
                print("Generation: ", i)
                print("Cohesion: ", mc)
                print("Alignment: ", ma)
                print("Fittest individual: ", fi)
                print("\n")

            self.pop = self.create_new_population(mutation_rate, mutation_step, tournament_size)
        
        # Run and evaluate one last time
        alignments, cohesions = sim.run_simulation(num_steps, show_sim=show_sim, log_states=log_states, filename=save_file_extension, log_dir=self.log_dir, last_gen=True)
        self.compute_fitness(alignments, cohesions)
                
        if log_perf:
            _, _, _ = track_performance.update(self.pop)
            performance = track_performance.get_performance()
            self.save_performance(performance, save_file_extension)

        if save_population:
            self.save_population(save_file_extension)

    def compute_fitness(self, alignments, cohesions):
        for i, agent in enumerate(self.pop):
            alignment = np.mean(alignments[i][-100:])   # Mean of last 100 alignment values for this agent
            cohesion = np.mean(cohesions[i][-100:])     # Mean of last 100 cohesion values for this agent
            cohesion_score = (cohesion**2) * 10         # Normalize cohesion score to the same range as alignment score
            
            agent["cohesion_score"] = cohesion_score
            agent["alignment_score"] = alignment
            agent["fitness"] = cohesion_score + alignment
    
    def create_new_population(self, mu, ms, ts):
        new_pop = []
        for id in range(len(self.pop)):
            
            # no crossover population
            if self.crossover_type == 'none':
                if ts == 1:
                    parent = self.fitness_proportionate_selection()
                else:
                    parent = self.tournament_selection(self.pop, ts)
                offspring = copy.deepcopy(parent)
                self.mutate(offspring["params"], mu, ms)
                offspring["id"] = id
            
            # crossover population
            else:
                if ts == 1:
                    parent1 = self.fitness_proportionate_selection()
                    parent2 = self.fitness_proportionate_selection()
                    # Ensure 2 different parents
                    while parent1['id'] == parent2['id']:
                        parent2 = self.fitness_proportionate_selection()
                else:
                    parent1 = self.tournament_selection(self.pop, ts)
                    parent2 = self.tournament_selection(self.pop, ts)
                    # Ensure 2 different parents
                    while parent1['id'] == parent2['id']:
                        parent2 = self.tournament_selection(self.pop, ts)
                
                offspring = self.crossover(parent1, parent2)
                self.mutate(offspring["params"], mu, ms)
                offspring["id"] = id
            new_pop.append(offspring)
        return new_pop

    def tournament_selection(self, pop, tournament_size):
        selection = np.random.choice(len(pop), size=tournament_size, replace=False) # Indices of individiuals in the tournament
        winner = selection[np.argmin([pop[i]["fitness"] for i in selection])] # Find index of fittest individual
        return pop[winner]
    
    def fitness_proportionate_selection(self):
        fitnesses = np.array([i["fitness"] for i in self.pop])
        scaled = -fitnesses + np.min(fitnesses) + np.max(fitnesses)
        probabilities = scaled / np.sum(scaled)
        return np.random.choice(self.pop, p=probabilities)

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_type == 'single_point':
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_type == 'mean':
            return self.mean_crossover(parent1, parent2)
        elif self.crossover_type == 'none':
            return self.no_crossover(parent1, parent2)

    def no_crossover(self, parent1, parent2):
        # Random sample between both parents
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

    def mutate(self, params, mutation_rate, mutation_step):
        w, b = params
        for weight in w:
            if np.random.random() < mutation_rate:
                weight += np.random.normal(0, mutation_step)

        for bias in b:
            if np.random.random() < mutation_rate:
                bias += np.random.normal(0, mutation_step)
        return w, b
    
    def save_performance(self, performance, save_file_extension): #TODO save more than fitness
        # Save performance over time dictionary of the population.
        save_to = self.log_dir + "/performance_" + save_file_extension + ".json"
        with open(save_to, "w") as jf:
            json.dump(performance, jf)
    
    def save_population(self, save_file_extension):
        # Save weights, biases and fitnesses of population
        weights = np.array([i["params"][0] for i in self.pop], dtype=object)
        biases = np.array([i["params"][1] for i in self.pop], dtype=object)
        fitnesses = np.array([i["fitness"] for i in self.pop], dtype=object)
        np.savez(self.log_dir + "/pop_info_" + save_file_extension, weights = weights, biases = biases, fitnesses = fitnesses)