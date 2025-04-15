import numpy as np
import utils

def main(activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs, mse_threshold):
    X_train, y_train, X_test, y_test = utils.preprocessing("mlp")
    weights, biases = train(X_train, y_train, activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs, mse_threshold)
    
    # Train and test accuracies
    train_tanh_confusion_matrix, train_tanh_accuracy_value = train_accuracy(X_train, y_train, weights, biases, bias, 'tanh')
    train_sigmoid_confusion_matrix, train_sigmoid_accuracy_value = train_accuracy(X_train, y_train, weights, biases, bias, 'sigmoid')
    test_tanh_confusion_matrix, test_tanh_accuracy_value = test_accuracy(X_test, y_test, weights, biases, bias, 'tanh')
    test_sigmoid_confusion_matrix, test_sigmoid_accuracy_value = test_accuracy(X_test, y_test, weights, biases, bias, 'sigmoid')
    
    print("Tanh Training Accuracy: {:.2f}%".format(train_tanh_accuracy_value))
    print("Tanh Test Accuracy: {:.2f}%".format(test_tanh_accuracy_value))
    
    print("Sigmoid Training Accuracy: {:.2f}%".format(train_sigmoid_accuracy_value))
    print("Sigmoid Test Accuracy: {:.2f}%".format(test_sigmoid_accuracy_value))

    print("Tanh Train Confusion Matrix:")
    print(train_tanh_confusion_matrix)
    print("Sigmoid Train Confusion Matrix:")
    print(train_sigmoid_confusion_matrix)
    
    print("Tanh Test Confusion Matrix:")
    print(test_tanh_confusion_matrix)
    print("Sigmoid Test Confusion Matrix:")
    print(test_sigmoid_confusion_matrix)

def train(X_train, y_train, activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs, mse_threshold):
    # Initialize weights and biases
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
        # Forward pass
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

        # Backpropagation
        error_signals = []
        error = y_train - layer_outputs[-1]
        current_error_signal = error * utils.activation_fn_derivative(layer_outputs[-1], activation_function)
        error_signals.append(current_error_signal)

        for i in range(len(weights) - 2, -1, -1):
            current_error_signal = np.dot(error_signals[0], weights[i + 1].T) * utils.activation_fn_derivative(layer_outputs[i + 1], activation_function)
            error_signals.insert(0, current_error_signal)

        # Update weights and biases
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

def test_accuracy(X_test, y_test, weights, biases, bias, activation_function):
    y_pred = predict(X_test, weights, biases, bias, activation_function)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    num_classes = y_test.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_test_classes, y_pred_classes):
        confusion_matrix[true, pred] += 1

    accuracy = np.mean(y_pred_classes == y_test_classes) * 100
    return confusion_matrix, accuracy

def train_accuracy(X_train, y_train, weights, biases, bias, activation_function):
    print("Confusion Matrix For Train Set using ", activation_function, " activation function")
    y_pred = predict(X_train, weights, biases, bias, activation_function)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)

    num_classes = y_train.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_train_classes, y_pred_classes):
        confusion_matrix[true, pred] += 1

    accuracy = np.mean(y_pred_classes == y_train_classes) * 100
    return confusion_matrix, accuracy

# Example of running the model with 'sigmoid' activation
activation_function = 'sigmoid'  # Try 'tanh' or 'sigmoid'
bias = True  # Set to False if no bias is needed
neurons_per_hidden_layer = 5  # Adjust as needed
hidden_layers = 2  # Adjust as needed
eta = 0.1  # Learning rate
epochs = 1000  # Number of epochs for training
mse_threshold = 0.01  # Mean squared error threshold for stopping criterion

# Run the main function
main(activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs, mse_threshold)
