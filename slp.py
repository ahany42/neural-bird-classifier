import utils
import numpy as np

def main(feature1, feature2, class1, class2, eta, epochs, bias):
    print("Single Layer Perceptron Training")
    print(feature1, feature2, class1, class2, eta, epochs, bias, sep="  ")
    X_train, y_train, X_test, y_test = utils.preprocessing(feature1, feature2, class1, class2)
    weights = train(X_train, y_train, eta, epochs, bias)
    y_pred = predict(X_test, weights, bias)
    print("True Labels (Encoded):", y_test)
    print("Predicted Labels (Encoded):", y_pred)
    accuracy ,TP,FP,FN,TN = utils.evaluate(y_test, y_pred)
    return accuracy ,TP,FP,FN,TN

def train(X_train, y_train, eta, epochs, bias):
    np.random.seed(42)
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64).flatten()  

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if bias:
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]

    weights = np.random.rand(X_train.shape[1])

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")  

    for epoch in range(epochs):
        for i in range(X_train.shape[0]):  
            net = np.dot(X_train[i], weights)
            y_pred = utils.signum_activation_fn(net)  

            target = y_train[i]
            error = target - y_pred  

            if error != 0:
                weights += eta * error * X_train[i]

    return weights

def predict(X_test, weights, bias):
    X_test = np.array(X_test, dtype=np.float64)
    y_pred = []

    if bias:
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    for i in range(len(X_test)):
        net = np.dot(X_test[i], weights)
        y_pred.append(utils.signum_activation_fn(net))

    return np.array(y_pred)  

def signum_activation_fn(x):
    return 1 if x > 0 else -1 if x < 0 else 0