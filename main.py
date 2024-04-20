from Evolutionary import EA

### Hyperparameters
pop_size = 40
num_generations = 10
mutation_rate = 0.05
tournament_size = 5

def main():
    pop_size = 100
    layer_sizes = [4, 3, 1]
    ea = EA(pop_size, layer_sizes)

    mutation_rate = 0.05
    tournament_size = 10
    ea.run(num_generations, mutation_rate, tournament_size, log=True)

main()
