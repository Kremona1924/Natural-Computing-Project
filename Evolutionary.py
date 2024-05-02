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

    def run(self, num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log = False, plot_chart=False, show_screen=True):
        for i in range(num_generations):
            sim = boids_sim(self.pop)
            agents = sim.run_with_screen(num_steps, plot_chart = plot_chart, show_screen=show_screen, rtrn=True, log=log, filename="simulation_log.json")
            self.evaluate_alignment(agents)
            self.evaluate_cohesion(agents)
            self.fitness(agents)
            self.set_scores(agents)

            if log:
                print("Generation: ", i)
                print("Fittest individual: ", min(self.pop, key=lambda x: x["fitness"])["fitness"])
            self.pop = self.create_new_population(mutation_rate, mutation_step, tournament_size)
        self.save_params(min(self.pop, key=lambda x: x["fitness"]))

    def evaluate_alignment(self, agents): # TODO alignment is only calculated from last recorded velocity of simulation, so maybe try last 10?
        global_alignment_x, global_alignment_y = self.get_global_alignment(agents)

        for agent in agents:
            agent_magnitude = math.sqrt(agent.xvel**2 + agent.yvel**2)
            xvel_norm = agent.xvel/agent_magnitude
            yvel_norm = agent.yvel/agent_magnitude

            agent.alignment_score = abs(xvel_norm-global_alignment_x) + abs(yvel_norm - global_alignment_y)
            
    
    def get_global_alignment(self, agents):
        # This is just a test but should probably be local (need neighbours and stuff for that)
        tot_xvel = 0
        tot_yvel = 0

        for agent in agents:
            agent_magnitude = math.sqrt(agent.xvel**2 + agent.yvel**2)
            tot_xvel += agent.xvel/agent_magnitude
            tot_yvel += agent.yvel/agent_magnitude
        
        return tot_xvel/len(agents), tot_yvel/len(agents)
    
    def evaluate_cohesion(self, agents):
        for agent in agents:
            # also needs neighbors
            neighbors = agent.get_neighbors(agents)
            # for now an if statement because not sure if we want to look at only local neighbors
            if neighbors:
                total_dist = 0
                for n in neighbors:
                    # distance to each neighbor
                    dist = math.sqrt((agent.xpos - n.xpos) ** 2 + (agent.ypos - n.ypos) ** 2)
                    total_dist += dist

                avg_dist = total_dist / len(neighbors)
                # use inverse since shorter distance means better cohesion
                agent.cohesion_score = 1 / (1 + avg_dist)  
            else:
                agent.cohesion_score = 0  # no cohesion if no neighbors


    def fitness(self, agents): #TODO change fitness function
        # Calculates fitness based on only alignment score 
        # Changed tournament criterion to argmin for now
        
        for agent in agents:
            fitness = agent.alignment_score + agent.cohesion_score
            agent.fitness = fitness

    def set_scores(self, agents):
        for i in range(len(agents)):
            self.pop[i]["fitness"] = agents[i].fitness


    def create_new_population(self, mu, ms, k):
        new_pop = []
        for id in range(len(self.pop)):
            parent = self.tournament_selection(self.pop, k, 1)
            offspring = copy.deepcopy(parent)
            self.mutate(offspring["params"], mu, ms)
            offspring["id"] = id
            # TODO: Add crossover
            new_pop.append(offspring)
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
    
    def crossover(self, x, y):
        return # TODO: Implement crossover

    def save_fitness(self, pop):
        # Save fitness values of the population. At each step, all fitness values are printed to one line
        pop_fitness = [i["fitness"] for i in pop]
        with open("fitness_over_time.txt", "a") as f:
            np.savetxt(f,pop_fitness, newline=' ')
            f.write("\n")
    
    def save_params(self, agent):
        # Save the parameters of the agent
        np.save('params.npy', np.array(agent["params"], dtype=object), allow_pickle=True)