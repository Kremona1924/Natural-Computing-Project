from Evolutionary import EA
import os

list_crossover = ['none', 'uniform', 'single_point', 'two_point']

### Hyperparameters
pop_size = 40
layer_sizes = [4, 4, 3, 1]
num_generations = 50
mutation_rate = 0.10
mutation_step = 0.10
tournament_size = 2
num_steps = 500
crossover_index = 3 # kies voor nu 0, 1, 2 of 3, later for-loop

def main():
    ea = EA(pop_size, layer_sizes, crossover_type = list_crossover[crossover_index])
    ea.run(num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log=True, plot_chart=False, show_screen=True, save_population=True)

main()
