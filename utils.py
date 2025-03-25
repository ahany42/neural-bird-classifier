import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def evaluate(class1, class2, y, y_pred):
    y_decoded = np.where(y == -1, class1, class2)
    y_pred_decoded = np.where(y_pred == -1, class1, class2)

    TP = np.sum((y_decoded == class1) & (y_pred_decoded == class1))
    FP = np.sum((y_decoded == class2) & (y_pred_decoded == class1))
    FN = np.sum((y_decoded == class1) & (y_pred_decoded == class2))
    TN = np.sum((y_decoded == class2) & (y_pred_decoded == class2))

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    return accuracy, TP, FP, FN, TN

def tanh_activation_fn(x):
    positive_expo = np.exp(x)
    negative_expo = np.exp(-x)
    return (positive_expo - negative_expo) / (positive_expo + negative_expo)

def linear_activation_fn(x):
    return x;

def sigmoid_activation_fn(x):
    return 1 / (1 + np.exp(-x))

def signum_activation_fn(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def preprocessing(feature1, feature2, class1, class2,algorithm):
    if algorithm == "slp" or algorithm == "Adaline":
        df = pd.read_csv('birds_data.csv')
        df["gender"].fillna(df["gender"].mode()[0], inplace=True)
        df["gender"] = df["gender"].astype("category").cat.codes  

        df_filtered = df[df["bird category"].isin([class1, class2])].copy()
        df_filtered = df_filtered[[feature1, feature2, "bird category"]]

        for feature in [feature1, feature2]:
            if df_filtered[feature].dtype == 'object':  
                df_filtered[feature] = df_filtered[feature].astype("category").cat.codes

        label_encoder = LabelEncoder()
        df_filtered["bird category"] = label_encoder.fit_transform(df_filtered["bird category"])
        df_filtered["bird category"] = df_filtered["bird category"].replace({0: -1, 1: 1})

        df_class1 = df_filtered[df_filtered["bird category"] == -1].sample(n=50, random_state=42)
        df_class2 = df_filtered[df_filtered["bird category"] == 1].sample(n=50, random_state=42)

        df_final = pd.concat([df_class1, df_class2])
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        X = df_final[[feature1, feature2]].values
        y = df_final["bird category"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)  

        return X_train, y_train, X_test, y_test
    elif algorithm == "mlp":
        #mlp pre processing here
        pass
    else:
        print("Invalid Algorithm")