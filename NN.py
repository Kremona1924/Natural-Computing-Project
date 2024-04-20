import numpy as np

def initialise_network(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        # TODO: better weight/bias initialisation
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
        biases.append(np.random.randn(layer_sizes[i+1]))
    return weights, biases

def feed_forward(params, inputs):
    weights, biases = params
    activations = inputs
    for i in range(len(weights) - 1):
        z = np.dot(activations, weights[i]) + biases[i]
        activations = np.maximum(0, z) # Relu activation
    z = np.dot(activations, weights[-1]) + biases[-1]
    return np.tanh(np.squeeze(z)) # Tanh output function
