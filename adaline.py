import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias):
    print("Adaline")
    print(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias, sep="  ")
    
    X_train, y_train, X_test, y_test = utils.preprocessing(feature1, feature2, class1, class2)
    
    weights = train(X_train, y_train, eta, epochs, mse_threshold, bias)
    
    predict(X_train, y_train, weights, bias)
    test()
    evaluate()

def train(X, y, eta, epochs, mse_threshold, bias):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # If X is 1D, reshape it (GUI Handling)
    if X.ndim == 1:  
        X = X.reshape(-1, 1)
    
    np.random.seed(42)
    
    weights = np.random.rand(X.shape[1] + int(bias))  
    
    for e in range(epochs):
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        for i in range(len(X)):
            x_i, y_i = X[i], y[i]
            
            net = np.dot(x_i, weights[1:]) + weights[0] if bias else np.dot(x_i, weights)
            net = utils.linear_activation_fn(net)
            error = y_i - net
            
            if not bias:
                weights += eta * error * x_i
            else:
                weights[1:] += eta * error * x_i
                weights[0] += eta * error
        
        mse = np.mean((y - np.dot(X, weights[1:]) - weights[0])**2) if bias else np.mean((y - np.dot(X, weights))**2)
        
        if mse < mse_threshold:
            break  
    
    return weights

def predict(X, y, weights, bias):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    
    if bias:
        y_vals = -(weights[1] * x_vals + weights[0]) / weights[2]
    else:
        y_vals = -(weights[0] * x_vals) / weights[1]
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

def test():
    pass
    
def evaluate():
    utils.confusion_matrix
    utils.accuracy_score
