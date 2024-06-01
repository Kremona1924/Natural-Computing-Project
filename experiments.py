from Evolutionary import EA, create_log_dir
import time

root = "logs/experiment"
experiment_dir = create_log_dir(dir=root)


### Fixed Hyperparameters
num_steps = 500
pop_size = 25
layer_sizes = [4, 4, 3, 1]
num_generations = 50
log_states = 'log_last' # log_last or log_all or log_none
log_performance = True

# Variable Hyperparameter Lists
mutation_rate_experiments = [0.01, 0.1, 0.3, 0.5]
mutation_step_experiments = [0.01, 0.1, 0.2, 0.4]
tournament_size_experiments = [1, 2, 4, 10]
crossover_type_experiments = ['none', 'uniform', 'single_point', 'mean']

start = time.time()
for mr in mutation_rate_experiments:
    run_dir = create_log_dir(dir=f"{experiment_dir}/run")

    mutation_rate = mr
    mutation_step = mutation_step_experiments[0]
    tournament_size = tournament_size_experiments[0]
    crossover_type = crossover_type_experiments[0]

    filename = f"steps{num_steps}_pop{pop_size}_gens{num_generations}_cot={crossover_type}_mr{mutation_rate:.0e}_ms{mutation_step:.0e}_ts{tournament_size}"

    # Run experiment
    print("------------------------------")
    print(f"Running experiment -> mr={mr}")
    print(". . .")
  
    ea = EA(pop_size, layer_sizes, crossover_type, run_dir)
    ea.run(num_generations, num_steps, mutation_rate, mutation_step, tournament_size, log_states=log_states, log_perf=log_performance, show_sim=False, save_population=True, save_file_extension=filename)


    print("Experiment complete")
    print("The files are saved at location: ", run_dir)
    end = time.time()
    print("Total running time (s): " + str(round(end-start)))

