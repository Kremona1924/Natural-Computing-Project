import glob
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import bisect
import itertools

mutation_rate_experiments = [0.01, 0.1, 0.3, 0.5]
mutation_step_experiments = [0.01, 0.1, 0.2, 0.4]
tournament_size_experiments = [1, 2, 4, 10]
crossover_type_experiments = ['none', 'uniform', 'single_point', 'mean']

def get_name(cot, mr, ms, sel):
    return f"performance_steps500_pop25_gens50_cot={cot}_mr{mr:.0e}_ms{ms:.0e}_{sel}.json"

def parse_name(name):
    mr = name.split("mr")[1].split("_")[0]
    ms = name.split("ms")[1].split("_")[0]
    cot = name.split("cot=")[1].split("_mr")[0]
    sel = name.split("ts")[-1].split(".")[0]
    return cot, float(mr), float(ms), int(sel)

run_folder = "logs/"

# data = {key : np.empty((64*5,51)) for key in tournament_size_experiments}
# counts = {key : 0 for key in tournament_size_experiments}

# for f1 in os.listdir(run_folder):
#     for f2 in os.listdir(run_folder + f1):
#         for f3 in os.listdir(run_folder + f1 + '/' + f2):
#             if f3[:11] == "performance":
#                 with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                     d = json.load(f)
#                     cot, mr, ms, sel = parse_name(f3)
#                     data[sel][counts[sel]] = d['mean_fitness_over_time']
#                     counts[sel] += 1

# for k, v in data.items():
#     plt.plot(np.mean(v, axis=0), label="Tournament size = " + str(k))

# plt.legend()
# plt.show()

# #======================================================
# N = 10
# best = [100] * N
# names = [None] * N

# for f1 in os.listdir(run_folder):
#     for f2 in os.listdir(run_folder + f1):
#         for f3 in os.listdir(run_folder + f1 + '/' + f2):
#             if f3[:11] == "performance":
#                 with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                     d = json.load(f)['mean_fitness_over_time']
#                     mean = np.mean(d[-10:])
#                     if mean < best[-1]:
#                         best += [mean]
#                         names += [f3]
#                         ind = np.argsort(best)
#                         best = [best[i] for i in ind][:-1]
#                         names = [names[i] for i in ind][:-1]

# print(best)
# [print(parse_name(n)) for n in names]


#=====================================================================

# mutation_step_experiments = mutation_step_experiments[::-1]
# data = {(mr, ms) : 0.0 for mr, ms in itertools.product(mutation_rate_experiments, mutation_step_experiments)}

# print(data.keys())
# for f1 in os.listdir(run_folder):
#     for f2 in os.listdir(run_folder + f1):
#         for f3 in os.listdir(run_folder + f1 + '/' + f2):
#             if f3[:11] == "performance":
#                 with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                     d = json.load(f)['mean_fitness_over_time']
#                     cot, mr, ms, sel = parse_name(f3)
#                     if cot == "single_point":
#                         mean = np.mean(d[-10:])
#                         data[(mr, ms)] += mean

# data_2d = np.zeros((len(mutation_rate_experiments), len(mutation_step_experiments)))
# for i, mr in enumerate(mutation_rate_experiments):
#     for j, ms in enumerate(mutation_step_experiments):
#         data_2d[i,j] = data[(mr, ms)]

# data_2d /= 4 * 5

# plt.imshow(data_2d, cmap='Blues', interpolation='nearest')
# plt.title("Effects of mutation rate and size\n(crossover type = uniform)")
# plt.xticks(range(len(mutation_rate_experiments)), mutation_rate_experiments)
# plt.yticks(range(len(mutation_step_experiments)), mutation_step_experiments)
# plt.xlabel("Mutation rate")
# plt.ylabel("Mutation step size")
# plt.colorbar(label="Mean fitness over last 10 generations")

# for (j,i),label in np.ndenumerate(data_2d):
#     plt.text(i,j,f"{label:.3f}",ha='center',va='center')

# plt.show()


#=====================================================================
data = {(mr, ms, cot, sel) : 0.0 for mr, ms, cot, sel in itertools.product(mutation_rate_experiments, mutation_step_experiments, crossover_type_experiments, tournament_size_experiments)}

for f1 in os.listdir(run_folder):
    for f2 in os.listdir(run_folder + f1):
        for f3 in os.listdir(run_folder + f1 + '/' + f2):
            if f3[:11] == "performance":
                with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
                    d = json.load(f)['mean_fitness_over_time']
                    cot, mr, ms, sel = parse_name(f3)
                    mean = np.mean(d[-10:])
                    data[(mr, ms, cot, sel)] += mean

for k in data.keys():
    data[k] /= 5

best_inds = np.argsort(list(data.values()))[:10]
for i in best_inds:
    print("Parameters:", list(data.keys())[i])