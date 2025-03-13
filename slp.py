import utils;
def main(feature1, feature2, class1, class2, eta, epochs, bias):
    print("slp")
    utils.preprocessing()
    train()
    predict()
    test()
    evaluate()
def train():
    pass

def predict():
    pass

def test(y,y_pred):
    error = y - y_pred;

#Call Accuracy And Confusion Matrix from utils
def evaluate():
    pass