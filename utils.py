import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def evaluate(y, y_pred):
    TP = sum((y == 1) & (y_pred == 1))
    FP = sum((y == 0) & (y_pred == 1))
    FN = sum((y == 1) & (y_pred == 0))
    TN = sum((y == 0) & (y_pred == 0))
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    return accuracy ,TP,FP,FN,TN

def linear_activation_fn(x):
    return x;

def signum_activation_fn(x):
    return 1 if x > 0 else 0  

def preprocessing(feature1, feature2, class1, class2):
    df = pd.read_csv('birds_data.csv')
    
    if feature1 not in df.columns or feature2 not in df.columns or "bird category" not in df.columns:
        raise KeyError(f"One or more columns ({feature1}, {feature2}, 'bird category') are missing in the dataset")
    
    df_filtered = df[df["bird category"].isin([class1, class2])]
    df_filtered = df_filtered[[feature1, feature2, "bird category"]].copy()

    for feature in [feature1, feature2]:
        if df_filtered[feature].dtype == 'object':  # Check if it's categorical
            df_filtered[feature] = df_filtered[feature].astype("category").cat.codes

    label_encoder = LabelEncoder()
    df_filtered["bird category"] = label_encoder.fit_transform(df_filtered["bird category"])
    print(df)

    X = df_filtered[[feature1, feature2]].values
    y = df_filtered["bird category"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test, label_encoder