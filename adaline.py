import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias):
    print("Adaline")
    print(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias, sep="  ")
    
    X_train, y_train, X_test, y_test = utils.preprocessing("Adaline",feature1, feature2, class1, class2)
    
    
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-6)
    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-6)
    
    weights = train(X_train, y_train, eta, epochs, mse_threshold, bias)
    
    y_pred= test(X_test,weights, bias)
    
    print("y_test:", y_test)
    print("y_pred:", y_pred)
    accuracy, TP, FP, FN, TN = utils.evaluate(class1, class2, y_test, y_pred)
    return accuracy, TP, FP, FN, TN
def test(X_test, weights, bias):
    y_pred = predict(X_test, weights, bias)
    return y_pred
def train(X, y, eta, epochs, mse_threshold, bias):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    if X.ndim == 1:  
        X = X.reshape(-1, 1)
    
    np.random.seed(42)
    weights = np.random.randn(X.shape[1] + int(bias)) * 0.01  

    for epoch in range(epochs):
        mse_total = 0

        for i in range(len(X)):  
            net_input = np.dot(X[i], weights[1:]) + weights[0] if bias else np.dot(X[i], weights)
            output = utils.activation_fn(net_input,"linear")
            error = y[i] - output
            
            if bias:
                weights[1:] += eta * error * X[i]
                weights[0] += eta * error
            else:
                weights += eta * error * X[i]

            mse_total += error**2
        
        mse = mse_total / len(X)

        if mse < mse_threshold:
            break  

    return weights

def predict(X, weights, bias):

    net_input = np.dot(X, weights[1:]) + weights[0] if bias else np.dot(X, weights)

    y_pred = np.array([utils.activation_fn(x,"signum") for x in net_input])

    plot_decision_boundary(X, y_pred, weights, bias)

    return y_pred


def plot_decision_boundary(X, y, weights, bias):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    
    if bias:
        y_vals = -(weights[1] * x_vals + weights[0]) / (weights[2] + 1e-6)
    else:
        y_vals = -(weights[0] * x_vals) / (weights[1] + 1e-6)

    plt.plot(x_vals, y_vals, 'k-', linewidth=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()
