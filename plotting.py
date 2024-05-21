from matplotlib import pyplot as plt
import numpy as np
import NN

def plot_network_outputs():
    pop_params = np.load('population_info.npz', allow_pickle=True)
    weigts = pop_params['weights']
    biases = pop_params['biases']
    outputs = []
    fig, axs = plt.subplots(3,3, squeeze=True)
    axs = axs.flatten()
    inputs = np.array([[0,0,0,0],
                       [1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1],
                       [1,1,1,1]])
    for i, input in enumerate(inputs):
        print(input)
        outputs = []
        for j in range(len(weigts)):
            outputs.append(NN.feed_forward((weigts[j], biases[j]), input))
        axs[i].hist(outputs)
        axs[i].set_xlim(-1,1)
    plt.show()

def plot_fitness_over_time():
    mins = np.empty((10,50))
    for i in range(3):
        pop_data = np.loadtxt('fitness_over_time_tour_k2' + str(i) + '.txt')
        mean = np.mean(pop_data, axis=1)
        min = np.min(pop_data, axis=1)
        mins[i] = min
        plt.plot(mean, label = "Run " + str(i), c = "r")
        plt.plot(min, label = "Run " + str(i), c = "k")
    #plt.plot(np.mean(mins, axis=0), c = "k")
    plt.show()

# plot_network_outputs()
plot_fitness_over_time()