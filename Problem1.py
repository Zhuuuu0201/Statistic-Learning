import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights():
    alpha1 = np.array([0.1, 0.3])
    alpha2 = np.array([0.3, 0.4])
    beta1 = np.array([0.4, 0.6])
    return alpha1, alpha2, beta1

def forward_propagation(x, alpha1, alpha2, beta1):
    hidden_input = np.dot(alpha1, x)
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(alpha2, hidden_output)
    network_output = sigmoid(output_input)

    return hidden_output, network_output

def backward_propagation(x, y, hidden_output, network_output, alpha1, alpha2, beta1, eta):
    error = y - network_output

    output_delta = error * sigmoid_derivative(network_output)

    hidden_error = np.dot(alpha2.T, output_delta)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights
    alpha2 += eta * output_delta * hidden_output
    alpha1 += eta * hidden_delta * x
    beta1 += eta * output_delta * hidden_output

    return alpha1, alpha2, beta1

def train_neural_network(xd, yd, alpha1, alpha2, beta1, eta, epochs):
    for epoch in range(epochs):
        hidden_output, network_output = forward_propagation(xd, alpha1, alpha2, beta1)
        alpha1, alpha2, beta1 = backward_propagation(xd, yd, hidden_output, network_output, alpha1, alpha2, beta1, eta)

        if epoch % 1000 == 0:
            error = 0.5 * np.sum((yd - network_output)**2)
            print(f"Epoch: {epoch}, Error: {error}")

    return alpha1, alpha2, beta1

# Test the algorithm
xd = np.array([0, 1])
yd = 1
eta = 0.5
epochs = 10000

alpha1, alpha2, beta1 = initialize_weights()
alpha1, alpha2, beta1 = train_neural_network(xd, yd, alpha1, alpha2, beta1, eta, epochs)

# Test the trained network
_, output = forward_propagation(xd, alpha1, alpha2, beta1)
print(f"Trained Output: {output}")
