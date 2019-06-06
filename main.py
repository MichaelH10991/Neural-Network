import numpy as np


def sigmoid(x):
    '''Our sigmoid normalization function'''
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Our training set of inputs
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

# Our expected outputs
training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# Initializing weights
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

for iteration in range(20000):
    # Calculate the output
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # error calculation and back propegation
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)

    # multiply input layer with our adjustments and sum
    synaptic_weights += np.dot(input_layer.T, adjustments)


print("Synaptic weights after training:")
print(synaptic_weights)
print("Outputs after training:")
print(outputs)
