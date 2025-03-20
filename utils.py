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

def linear_activation_fn(x):
    return x;

def signum_activation_fn(x):
    return 1 if x > 0 else -1

def preprocessing(feature1, feature2, class1, class2):
    df = pd.read_csv('birds_data.csv')
    df["gender"].fillna(df["gender"].mode()[0], inplace=True)
    df["gender"] = df["gender"].astype("category").cat.codes  
    if feature1 not in df.columns or feature2 not in df.columns or "bird category" not in df.columns:
        raise KeyError(f"One or more columns ({feature1}, {feature2}, 'bird category') are missing in the dataset")
    
    df_filtered = df[df["bird category"].isin([class1, class2])].copy()
    df_filtered = df_filtered[[feature1, feature2, "bird category"]]

    for feature in [feature1, feature2]:
        if df_filtered[feature].dtype == 'object':  
            df_filtered[feature] = df_filtered[feature].astype("category").cat.codes

    label_encoder = LabelEncoder()
    df_filtered["bird category"] = label_encoder.fit_transform(df_filtered["bird category"])

    # Convert labels from (0,1) to (-1,1)
    df_filtered["bird category"] = df_filtered["bird category"].replace({0: -1, 1: 1})

    X = df_filtered[[feature1, feature2]].values
    y = df_filtered["bird category"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test