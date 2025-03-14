import utils;
def main(feature1, feature2, class1, class2, eta, epochs, bias):
    print("slp")
    print(feature1,feature2,class1,class2,eta,epochs,bias,sep="  ")
    X_train, y_train, X_test, y_test = utils.preprocessing(feature1, feature2, class1, class2)
    train()
    y_pred = predict()
    test(y_test,y_pred)
    evaluate()
def train():
    pass

def predict():
    y_pred =[]
    return y_pred;

def test(y,y_pred):
    error = y - y_pred;

#Call Accuracy And Confusion Matrix from utils
def evaluate():
    pass