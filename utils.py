import pandas as pd
def confusion_matrix():
    pass

def accuracy_score(y,y_pred):
    correct = 0
    total = len(y)
    for i in range(total):
        if y == y_pred:
            correct += 1
    return correct/total * 100;
def train_test_samples():
   pass

def linear_activation_fn(x):
    return x;

def signum_activation_fn(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def encoding():
    # from c1 , c2 to 1 and -1 or 0 1 according to activation fn
    pass

def input_output_mapping():
    #from labels to c1 c2 again
    pass