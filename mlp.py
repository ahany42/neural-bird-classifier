import numpy as np
import pandas as pd
import utils
from tqdm import tqdm
def main(activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs):
    X_train, y_train, X_test, y_test = utils.preprocessing("mlp")

    if isinstance(neurons_per_hidden_layer, str):
        neurons_per_hidden_layer = list(map(int, neurons_per_hidden_layer.split(',')))
    
    if len(neurons_per_hidden_layer) > hidden_layers:
        neurons_per_hidden_layer = neurons_per_hidden_layer[:hidden_layers]
    elif len(neurons_per_hidden_layer) < hidden_layers:
        neurons_per_hidden_layer += [neurons_per_hidden_layer[-1]] * (hidden_layers - len(neurons_per_hidden_layer))

    weights, biases = train(X_train, y_train, activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs)
    
    train_confusion_matrix, train_accuracy = accuracy(X_train, y_train, weights, biases, bias, activation_function)
    test_confusion_matrix, test_accuracy = accuracy(X_test, y_test, weights, biases, bias, activation_function)
    overall_accuracy = (train_accuracy + test_accuracy) / 2
   
    
    print(f"\nTraining Results with {activation_function}:")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print("Training Confusion Matrix:")
    print(train_confusion_matrix)
    
    print(f"\nTest Results with {activation_function}:")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)


    return train_confusion_matrix, test_confusion_matrix, train_accuracy, test_accuracy, overall_accuracy

def train(X_train, y_train, activation_function, bias, neurons_per_hidden_layer, hidden_layers, eta, epochs):
    weights = []
    biases = []

    input_size = X_train.shape[1]
    #small random weights and biases
    weights.append(np.random.randn(input_size, neurons_per_hidden_layer[0]) * 0.01)
    biases.append(np.random.randn(1, neurons_per_hidden_layer[0]) * 0.01 if bias else np.zeros((1, neurons_per_hidden_layer[0])))

    for i in range(hidden_layers - 1):
        weights.append(np.random.randn(neurons_per_hidden_layer[i], neurons_per_hidden_layer[i + 1]))
        biases.append(np.random.randn(1, neurons_per_hidden_layer[i + 1]) if bias else np.zeros((1, neurons_per_hidden_layer[i + 1])))

    output_size = y_train.shape[1]
    weights.append(np.random.randn(neurons_per_hidden_layer[-1], output_size))
    biases.append(np.random.randn(1, output_size) if bias else np.zeros((1, output_size)))


    for epoch in tqdm(range(epochs), desc="Training epochs"):
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
        # Back propagation
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

def accuracy(X, y, weights, biases, bias, activation_function):
    y_pred = predict(X, weights, biases, bias, activation_function)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y, axis=1)

    unique_classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
    num_classes = len(unique_classes)
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    df = pd.read_csv('birds_data.csv')
    class_names = df['bird category'].unique()
    labels = [class_names[idx_to_class[i]] for i in range(num_classes)]
    
    for true, pred in zip(y_true_classes, y_pred_classes):
        true_idx = class_to_idx[true]
        pred_idx = class_to_idx[pred]
        confusion_matrix[true_idx, pred_idx] += 1

    accuracy_value = np.mean(y_pred_classes == y_true_classes) * 100

    confusion_matrix = pd.DataFrame(confusion_matrix, 
                                  index=[f'Actual {label} ' for label in labels],
                                  columns=[f'{label}' for label in labels])

    return confusion_matrix, accuracy_value
