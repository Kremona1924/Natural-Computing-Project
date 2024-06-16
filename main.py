import time
import numpy as np

from Evolutionary import EA, create_log_dir

'''
This script runs the Evolutionary Algorithm for 1 experiment with changable hyperparameters. You can see the progression by 
looking at the simulation in pygame. Generally, a good solution is found with the following hyperparameters:

mutation_rate = 0.10
mutation_step = 0.20
tournament_size = 5
crossover_type = 'uniform'
'''

list_crossover = ['none', 'uniform', 'single_point', 'mean']

# Fixed Hyperparameters
num_steps = 500
pop_size = 25
layer_sizes = [4, 4, 3, 2]
num_generations = 50

log_states = 'log_last'     # Logs position in each simulation. Options: log_last, log_all, log_none
log_performance = True      # Logs performance such as fitness, cohesion and alignment over time
log_parameters = True       # Save weights and biases of the population after the last simulation

show_sim = True             # Shows the simulations in pygame

# Variable Hyperparameters
mutation_rate = 0.10
mutation_step = 0.20
tournament_size = 5
crossover_type = 'uniform' # Options: 'none', 'uniform', 'single_point', 'mean'

log_dir = create_log_dir("logs/single_experiment", suffix_len=4)
print("The files will be saved at location: ", log_dir)

filename = f"steps{num_steps}_pop{pop_size}_gens{num_generations}_cot={crossover_type}_mr{mutation_rate:.0e}_ms{mutation_step:.0e}_ts{tournament_size}"
print("Under the extension name: ", filename)

np.random.seed(11)

def main():
    ea = EA(pop_size, layer_sizes, crossover_type, log_dir=log_dir)
    ea.run(num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log_states, log_performance, show_sim, log_parameters, filename)

start = time.time()
main()
end = time.time()
print("It took " + str(round(end-start)) + " seconds to run")
