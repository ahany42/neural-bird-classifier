import utils
import numpy as np
import pandas as pd
def main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias):
    print("Adaline")
    print(feature1,feature2,class1,class2,eta,epochs,mse_threshold,bias,sep="  ")
    X_train, y_train, X_test, y_test = utils.preprocessing(feature1, feature2, class1, class2)
    weights = train(X_train, y_train, eta, epochs, mse_threshold, bias)
    predict()
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
        net = np.dot(X, weights) if not bias else np.dot(X, weights[1:]) + weights[0]
        net = utils.linear_activation_fn(net)  
        errors = y - net
        mse = np.mean(errors**2)
        
        if mse < mse_threshold:
            break
        
        if not bias:
            weights += eta * np.dot(X.T, errors) 
        else:
            weights[1:] += eta * np.dot(X.T, errors)  
        
        if bias:
            weights[0] += eta * errors.sum() 
    
    return weights
    
def predict():
    pass

def test():
    pass
    
def evaluate():
    utils.confusion_matrix
    utils.accuracy_score
