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
    np.random.seed(42)
    weights = np.random.rand(X.shape[1] + int(bias))  
    
    for epoch in range(epochs):
        output = np.dot(X, weights[1:]) + (weights[0] if bias else 0)  
        errors = y - output
        mse = np.mean(errors**2)
        
        if mse < mse_threshold:
            break
        
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
    
#testing train function
def unit_test_train():  
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
    y = np.array([0, 0, 0, 1]) 
    eta = 0.1 
    epochs = 1000  
    mse_threshold = 0.01  
    bias = False
    weights = train(X, y, eta, epochs, mse_threshold, bias)
    print("Final Weights:", weights)

# unit_test_train()