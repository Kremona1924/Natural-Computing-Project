from Evolutionary import EA

### Hyperparameters
num_generations = 100
pop_size = 1000
mutation_rate = 0.05
tournament_size = 50

def main():
    pop_size = 100
    layer_sizes = [2, 3, 1]
    ea = EA(pop_size, layer_sizes)

    mutation_rate = 0.05
    tournament_size = 10
    ea.run(num_generations, mutation_rate, tournament_size, log=True)

main()
