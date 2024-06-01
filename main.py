from Evolutionary import EA, create_log_dir
import os
import time

list_crossover = ['none', 'uniform', 'single_point', 'mean']

# Fixed Hyperparameters
num_steps = 500
pop_size = 25
layer_sizes = [4, 4, 3, 1]
num_generations = 50
log_states = 'log_last' # log_last or log_all or log_none
log_performance = True

# Variable Hyperparameters
mutation_rate = 0.10
mutation_step = 0.10
tournament_size = 2
crossover_index = 0 # kies voor nu 0, 1, 2 of 3, later for-loop

log_dir = create_log_dir()
print("The files will be saved at location: ", log_dir)

filename = f"steps{num_steps}_pop{pop_size}_gens{num_generations}_cot={list_crossover[crossover_index]}_mr{mutation_rate:.0e}_ms{mutation_step:.0e}_ts{tournament_size}"
print("Under the extension name: ", filename)


def main():
    ea = EA(pop_size, layer_sizes, crossover_type = list_crossover[crossover_index], log_dir=log_dir)
    ea.run(num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log_states=log_states, log_perf=log_performance, show_sim=False, save_population=True, save_file_extension=filename)

start = time.time()
main()
end = time.time()
print("It took " + str(round(end-start)) + " seconds to run")
