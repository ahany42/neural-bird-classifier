import pandas as pd
import random
def evaluate(class_one, class_two, y, y_pred):
    confusion_matrix(class_one, class_two, y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.2f}%")

def confusion_matrix(class_one,class_two,y,y_pred):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for t,prediction in zip(y,y_pred):
        if (t == class_one and prediction == class_one):
            TP +=1
        elif (t == class_one and prediction == class_two):
            FN +=1
        elif (t== class_two and prediction == class_two):
            TN +=1
        elif (t == class_two and prediction == class_one):
            FP +=1 
    print("TP ",TP)
    print("FP ",FP)
    print("FN ",FN)
    print("TN ",TN)

def accuracy_score(y,y_pred):
    correct = 0
    total = len(y)
    for i in range(total):
        if y [i]== y_pred[i]:
            correct += 1
    return correct/total * 100;

def linear_activation_fn(x):
    return x;

def signum_activation_fn(x):
    return 1 if x > 0 else -1 if x < 0 else 0
def mse():
    pass
def preprocessing(feature1, feature2, class1, class2):
    #call train test split
      X_train = [] 
      y_train = []
      X_test = []
      y_test = []
      print("PreProcessing ",feature1,feature2,class1,class2)
      return X_train, y_train, X_test, y_test