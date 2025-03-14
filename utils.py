import pandas as pd
import random
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

def accuracy_score(y,y_pred):
    correct = 0
    total = len(y)
    for i in range(total):
        if y == y_pred:
            correct += 1
    return correct/total * 100;

def two_classes_train_test_samples(df,class_one,class_two):
    df = df[df['bird category'].isin([class_one,class_two])].copy()
    train_data = []
    test_data = []
    class_one_samples = []
    class_two_samples = []
    dataframe_list = df.to_dict(orient="records")
    random.shuffle(dataframe_list)
    for row in dataframe_list:
        if row['bird category'] == class_one:
            class_one_samples.append(row)
        else:
            class_two_samples.append(row)
            
    for i in range(30):
        train_data.append(class_one_samples[i])
        train_data.append(class_two_samples[i])

    for i in range(30, 50):
        test_data.append(class_one_samples[i])
        test_data.append(class_two_samples[i])
        
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df
            

def linear_activation_fn(x):
    return x;

def signum_activation_fn(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def label_encoding(df,class_one,class_two):
    label_mapping = {class_one: -1, class_two: 1}
    encoded_labels = []
    for class_label in df['bird category']:
        encoded_labels.append(label_mapping[class_label])
    df['bird category'] = encoded_labels
    return df

def input_output_mapping(df,class_one,class_two):
    reverse_mapping = {-1: class_one, 0: class_two}
    decoded_labels = []
    for encoded_label in df['bird category']:
        decoded_labels.append(reverse_mapping[encoded_label])
    df['bird category'] = decoded_labels
    return df

def preprocessing(feature1, feature2, class1, class2):
      X_train = [] 
      y_train = []
      X_test = []
      y_test = []
      print("PreProcessing ",feature1,feature2,class1,class2)
      return X_train, y_train, X_test, y_test