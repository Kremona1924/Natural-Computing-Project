import glob
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import bisect
import itertools

mutation_rate_experiments = [0.01, 0.1, 0.2, 0.4]
mutation_step_experiments = [0.01, 0.1, 0.2, 0.4]
tournament_size_experiments = [1, 2, 5, 12]

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

# # #=======================================================================================
# # Plot fitness over time per cot and ts

# fig, axs = plt.subplots(1,4, sharey=True, figsize=(16,4))

# cots = ['none', 'uniform', 'single_point', 'mean']
# titles = ['No crossover', 'Uniform crossover', 'Single point crossover', 'Mean crossover']
# for i in range(4):
#     data = {key : np.empty((5,51)) for key in tournament_size_experiments}
#     counts = {key : 0 for key in tournament_size_experiments}

#     for f1 in os.listdir(run_folder):
#         for f2 in os.listdir(run_folder + f1):
#             for f3 in os.listdir(run_folder + f1 + '/' + f2):
#                 if f3[:11] == "performance":
#                     with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                         d = json.load(f)
#                         cot, mr, ms, sel = parse_name(f3)
#                         if mr == 0.2 and ms == 0.2:
#                             if cot == cots[i]:
#                                 data[sel][counts[sel]] = d['mean_fitness_over_time']
#                                 counts[sel] += 1

#     for k, v in data.items():
#         l = "Tournament (K = " + str(k) +")" if k != 1 else "Fitness proportionate"
#         axs[i].plot(np.mean(v, axis=0), label= l)
#         axs[i].set_title(titles[i])

# for ax in axs.ravel():
#     ax.set_xlabel("Generation")
#     ax.set_ylabel("Fitness")
#     ax.label_outer()
#     ax.legend()

# plt.show()

# # #=======================================================================================
# # Plot diversity over time

# fig, axs = plt.subplots(1,4, sharey=True, figsize=(16,4))

# run_folder = "logs_div_v2/"
# cots = ['none', 'uniform', 'single_point', 'mean']
# titles = ['No crossover', 'Uniform crossover', 'Single point crossover', 'Mean crossover']
# #for n, p in enumerate(['turn', 'speed']):
# for i in range(4):
#     data = {key : np.empty((5,51)) for key in tournament_size_experiments}
#     counts = {key : 0 for key in tournament_size_experiments}

#     for f1 in os.listdir(run_folder):
#         for f2 in os.listdir(run_folder + f1):
#             for f3 in os.listdir(run_folder + f1 + '/' + f2):
#                 if f3[:11] == "performance":
#                     with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                         d = json.load(f)
#                         cot, mr, ms, sel = parse_name(f3)
#                         if mr == 0.2 and ms == 0.2:
#                             if cot == cots[i]:
#                                 data[sel][counts[sel]] = (np.array(d['diversity_over_time']['turn']) + np.array(d['diversity_over_time']['speed']))/2.0
#                                 counts[sel] += 1

#     for k, v in data.items():
#         l = "Tournament (K = " + str(k) +")" if k != 1 else "Fitness proportionate"
#         axs[i].plot(np.mean(v, axis=0), label= l)
#         axs[i].set_title(titles[i])
#         axs[i].set_xlabel("Generation")
#         axs[i].set_ylabel("Diversity")# (turn)" if n == 0 else "Diversity (speed)")
#         axs[i].set_ylim(0,0.8)
#         axs[i].plot(0.102 * np.ones(51), 'k--')

# for ax in axs.ravel():
    
#     ax.label_outer()
#     ax.legend()

# plt.show()

# # #======================================================
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
#                         names += [f1 + '/' + f2 + '/' + f3]
#                         ind = np.argsort(best)
#                         best = [best[i] for i in ind][:-1]
#                         names = [names[i] for i in ind][:-1]

# print(best)
# [print(parse_name(n)) for n in names]
# print(names)

#=====================================================================
# # Plot mean fitness per mutation size and rate for each crossover type

# titles = ['No crossover', 'Uniform crossover', 'Single point crossover', 'Mean crossover']
# mutation_step_experiments = mutation_step_experiments[::-1]

# fig, axs = plt.subplots(1,4, sharey=True, figsize=(16,4))
# for i, c in enumerate(crossover_type_experiments):
#     data = {(mr, ms) : 0.0 for mr, ms in itertools.product(mutation_rate_experiments, mutation_step_experiments)}
#     for f1 in os.listdir(run_folder):
#         for f2 in os.listdir(run_folder + f1):
#             for f3 in os.listdir(run_folder + f1 + '/' + f2):
#                 if f3[:11] == "performance":
#                     with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                         d = json.load(f)['mean_fitness_over_time']
#                         cot, mr, ms, sel = parse_name(f3)
#                         if cot == c:
#                             mean = np.mean(d[-10:])
#                             data[(mr, ms)] += mean

#     data_2d = np.zeros((len(mutation_rate_experiments), len(mutation_step_experiments)))
#     for k, mr in enumerate(mutation_rate_experiments):
#         for j, ms in enumerate(mutation_step_experiments):
#             data_2d[k,j] = data[(mr, ms)]

#     data_2d /= 5 * 4

#     img = axs[i].imshow(data_2d, cmap='Reds', interpolation='nearest', vmin=0.66, vmax=1.05)
#     axs[i].set_title(titles[i])
#     axs[i].set_xticks(range(len(mutation_rate_experiments)), mutation_rate_experiments)
#     axs[i].set_yticks(range(len(mutation_step_experiments)), mutation_step_experiments)
#     axs[i].set_xlabel("Mutation rate")
#     axs[i].set_ylabel("Mutation size")
#     axs[i].label_outer()

#     for (j,k),label in np.ndenumerate(data_2d):
#         axs[i].text(k,j,f"{label:.3f}",ha='center',va='center')

# # cbar = fig.colorbar(img, ax=axs.ravel().tolist(), shrink=0.7, label="Mean fitness in last 10 generations")

# plt.savefig("mutation_rate_vs_mutation_step.svg")
# plt.show()


#=====================================================================
# # List best parameter configs
# data = {(mr, ms, cot, sel) : 0.0 for mr, ms, cot, sel in itertools.product(mutation_rate_experiments, mutation_step_experiments, crossover_type_experiments, tournament_size_experiments)}

# for f1 in os.listdir(run_folder):
#     for f2 in os.listdir(run_folder + f1):
#         for f3 in os.listdir(run_folder + f1 + '/' + f2):
#             if f3[:11] == "performance":
#                 with open(run_folder + f1 + '/' + f2 + '/' + f3, 'r') as f:
#                     d = json.load(f)['mean_fitness_over_time']
#                     cot, mr, ms, sel = parse_name(f3)
#                     if cot == 'none' and mr==0.2 and ms==0.2 and sel == 5:
#                         print(run_folder + f1 + '/' + f2 + '/' +f3)
#                     mean = np.mean(d[-20:])
#                     data[(mr, ms, cot, sel)] += mean

# for k in data.keys():
#     data[k] /= 5

# best_inds = np.argsort(list(data.values()))[:20]
# for i in best_inds:
#     print("Parameters:", list(data.keys())[i], "Mean fitness:", list(data.values())[i])