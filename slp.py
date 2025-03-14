import utils;
def main(feature1, feature2, class1, class2, eta, epochs, bias):
    print("slp")
    utils.preprocessing(feature1, feature2, class1, class2)
    train()
    y,y_pred = predict()
    test(y,y_pred)
    evaluate()
def train():
    pass

def predict():
    y,y_pred =[]
    return y,y_pred;

def test(y,y_pred):
    error = y - y_pred;

#Call Accuracy And Confusion Matrix from utils
def evaluate():
    pass