import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix, accuracy_score
import utils 

def train(X_train, y_train, activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs, mse_threshold):
    weights = []
    biases = []

    if isinstance(neurons_per_hidden_layer, int):
        neurons_per_hidden_layer = [neurons_per_hidden_layer] * hidden_layers

    input_size = X_train.shape[1]
    weights.append(np.random.randn(input_size, neurons_per_hidden_layer[0]))
    biases.append(np.random.randn(1, neurons_per_hidden_layer[0]) if bias else np.zeros((1, neurons_per_hidden_layer[0])))

    for i in range(hidden_layers - 1):
        weights.append(np.random.randn(neurons_per_hidden_layer[i], neurons_per_hidden_layer[i + 1]))
        biases.append(np.random.randn(1, neurons_per_hidden_layer[i + 1]) if bias else np.zeros((1, neurons_per_hidden_layer[i + 1])))

    output_size = y_train.shape[1]
    weights.append(np.random.randn(neurons_per_hidden_layer[-1], output_size))
    biases.append(np.random.randn(1, output_size) if bias else np.zeros((1, output_size)))

    for epoch in range(epochs):
        layer_inputs = [X_train]
        layer_outputs = [X_train]

        for i in range(len(weights)):
            net = np.dot(layer_outputs[-1], weights[i])
            if bias:
                net += biases[i]
            output = utils.activation_fn(net, activation_function)
            layer_inputs.append(net)
            layer_outputs.append(output)

        mse = np.mean((y_train - layer_outputs[-1]) ** 2)
        if mse < mse_threshold:
            break

        error_signals = []
        error = y_train - layer_outputs[-1]
        current_error_signal = error * utils.activation_fn_derivative(layer_outputs[-1], activation_function)
        error_signals.append(current_error_signal)

        for i in range(len(weights) - 2, -1, -1):
            current_error_signal = np.dot(error_signals[0], weights[i + 1].T) * utils.activation_fn_derivative(layer_outputs[i + 1], activation_function)
            error_signals.insert(0, current_error_signal)

        for i in range(len(weights)):
            weights[i] += eta * np.dot(layer_outputs[i].T, error_signals[i])
            if bias:
                biases[i] += eta * np.sum(error_signals[i], axis=0, keepdims=True)

    return weights, biases

def predict(X, weights, biases, bias, activation_function):
    layer_outputs = [X]

    for i in range(len(weights)):
        net = np.dot(layer_outputs[-1], weights[i])
        if bias:
            net += biases[i]
        output = utils.activation_fn(net, activation_function)
        layer_outputs.append(output)

    return layer_outputs[-1]

def test(X_test, y_test, weights, biases, bias, activation_function):
    y_pred = predict(X_test, weights, biases, bias, activation_function)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    num_classes = y_test.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_test_classes, y_pred_classes):
        confusion_matrix[true, pred] += 1

    accuracy = np.mean(y_pred_classes == y_test_classes) * 100
    return confusion_matrix, accuracy

# Main function
def main(eta, epochs, bias, neurons_per_hidden_layer, hidden_layers, activation_function, mse_threshold):
    # Get the dataset from utils
    X_train, y_train, X_test, y_test = utils.preprocessing("mlp")
    
    # Print dataset shapes for debugging
    print("\nDataset Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Convert neurons_per_hidden_layer to list if it's an integer
    if isinstance(neurons_per_hidden_layer, int):
        neurons_per_hidden_layer = [neurons_per_hidden_layer] * hidden_layers
    
    # Train the network
    weights, biases = train(X_train, y_train, activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs, mse_threshold)
    
    # Test the network
    confusion_matrix, accuracy = test(X_test, y_test, weights, biases, bias, activation_function)
    
    # Print results
    print("\nMLP Results:")
    print("------------")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nNetwork Configuration:")
    print(f"Learning Rate (eta): {eta}")
    print(f"Epochs: {epochs}")
    print(f"Bias: {'Enabled' if bias else 'Disabled'}")
    print(f"Neurons per Hidden Layer: {neurons_per_hidden_layer}")
    print(f"Number of Hidden Layers: {hidden_layers}")
    print(f"Activation Function: {activation_function}")
    print(f"MSE Threshold: {mse_threshold}")
    
    return confusion_matrix, accuracy


