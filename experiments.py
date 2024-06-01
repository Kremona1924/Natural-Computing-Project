from Evolutionary import EA, create_log_dir
import time
import random




# Fixed Hyperparameters
num_steps = 500
pop_size = 25
layer_sizes = [4, 4, 3, 2]
num_generations = 50
log_states = 'log_last' # log_last or log_all or log_none
log_performance = True
show_sim = False
save_population = True

# Variable Hyperparameter Lists
number_of_runs = 5  # Number of times to run each experiment configuration
random_seeds = [10, 20, 30, 40, 50]
mutation_rate_experiments = [0.01, 0.1, 0.3, 0.5]
mutation_step_experiments = [0.01, 0.1, 0.2, 0.4]
tournament_size_experiments = [1, 2, 4, 10]
crossover_type_experiments = ['none', 'uniform', 'single_point', 'mean']



start = time.time()

for run in range(number_of_runs):
    # Generate random seed from list
    random.seed(random_seeds[run])

    # Set run directory
    root = "logs/official_run"
    experiment_dir = create_log_dir(dir=root)

    # Start experiments
    print("------------------------------")
    print(f"Starting run {run+1}")
    for mr in mutation_rate_experiments:
        for ms in mutation_step_experiments:
            for ts in tournament_size_experiments:
                for ct in crossover_type_experiments:
                    #Create experiment directory
                    run_dir = create_log_dir(dir=f"{experiment_dir}/experiment", suffix_len=4)
                    
                    # Set file name based on hyperparameters
                    filename = f"steps{num_steps}_pop{pop_size}_gens{num_generations}_cot={ct}_mr{mr:.0e}_ms{ms:.0e}_ts{ts}"

                    # Run experiment
                    print("------------------------------")
                    print(f"Running experiment -> mr={mr}, ms={ms}, ts={ts}, ct={ct}")
                    print(". . .")
                
                    ea = EA(pop_size, layer_sizes, ct, run_dir)
                    ea.run(num_generations, num_steps, mr, ms, ts, log_states, log_performance, show_sim, save_population, filename)

                    print("Experiment complete")
                    print("The files are saved at location: ", run_dir)
                    end = time.time()
                    print("Total running time (s): " + str(round(end-start)))

