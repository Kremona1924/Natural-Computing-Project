from matplotlib import pyplot as plt
import numpy as np
import NN

def plot_network_outputs():
    pop_params = np.load('final_pop_params.npz', allow_pickle=True)
    weigts = pop_params['weights']
    biases = pop_params['biases']
    outputs = []
    for i in range(len(weigts)):
       outputs.append(NN.feed_forward((weigts[i], biases[i]), np.array([1,0,0,0])))
    plt.hist(outputs)
    plt.show()

plot_network_outputs()