import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias):
    print("Adaline")
    print(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias, sep="  ")
    
    X_train, y_train, X_test, y_test = utils.preprocessing(feature1, feature2, class1, class2)
    
    
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-6)
    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-6)
    
    weights = train(X_train, y_train, eta, epochs, mse_threshold, bias)
    
    y_pred = predict(X_test, weights, bias)
    
    print("y_test:", y_test)
    print("y_pred:", y_pred)
    accuracy, TP, FP, FN, TN = utils.evaluate(class1, class2, y_test, y_pred)
    return accuracy, TP, FP, FN, TN

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
            output = utils.linear_activation_fn(net_input)
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
    y_pred = np.where(net_input >= 0, 1, -1) 
    
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

#def test(X_test, y_test, weights, bias):
    #y_pred = predict(X_test, weights, bias)
    #print("Test Predictions:", y_pred)
    #return y_pred


# accuracy, TP, FP, FN, TN = main(
#     feature1="beak_length",
#     feature2="gender",
#     class1="A",
#     class2="B",
#     eta=0.01,
#     epochs=1000,
#     mse_threshold=4,  # Adjust as needed
#     bias=False  # This is a boolean
# )
# accuracy, TP, FP, FN, TN = main(
#     feature1="beak_length",
#     feature2="gender",
#     class1="A",
#     class2="B",
#     eta=0.01,
#     epochs=1000,
#     mse_threshold=4,  # Adjust as needed
#     bias=False 
# )

# print(f"Accuracy: {accuracy}%")
# print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
