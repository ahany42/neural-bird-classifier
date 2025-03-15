import utils
import numpy as np
from sklearn.linear_model import Perceptron
def main(feature1, feature2, class1, class2, eta, epochs, bias):
    print("slp")
    print(feature1, feature2, class1, class2, eta, epochs, bias, sep="  ")
    X_train, y_train, X_test, y_test = utils.preprocessing(feature1, feature2, class1, class2)
    weights = train(X_train, y_train, eta, epochs, bias)
    y_pred = predict(X_test, weights, bias)
    utils.evaluate(class1, class2, y_test, y_pred)

def train(X_train, y_train, eta, epochs, bias):
    np.random.seed(42)
    
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    if bias:
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    weights = np.random.rand(X_train.shape[1])  

    for e in range(epochs):
        for i in range(len(X_train)):
            net = np.dot(X_train[i], weights)
            y_pred = utils.signum_activation_fn(net)
            error = y_train[i] - y_pred
            if error != 0:
                weights += eta * error * X_train[i] 

    return weights

def predict(X_test, weights, bias):
    X_test = np.array(X_test, dtype=np.float64)
    if bias:
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
    y_pred = [utils.signum_activation_fn(np.dot(X_test[i], weights)) for i in range(len(X_test))]
    return y_pred

