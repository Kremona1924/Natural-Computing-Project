from Evolutionary import EA
import os

### Hyperparameters
pop_size = 50
layer_sizes = [4, 4, 3, 1]
num_generations = 100
mutation_rate = 0.10
mutation_step = 0.10
tournament_size = 2
num_steps = 500

def main():
    ea = EA(pop_size, layer_sizes)
    ea.run(num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log=True, plot_chart=False, show_screen=True, save_population=True)

main()
