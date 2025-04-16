import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def evaluate(class1, class2, y, y_pred):
    y_decoded = np.where(y == -1, class1, class2)
    y_pred_decoded = np.where(y_pred == -1, class1, class2)

    TP = np.sum((y_decoded == class1) & (y_pred_decoded == class1))
    FP = np.sum((y_decoded == class2) & (y_pred_decoded == class1))
    FN = np.sum((y_decoded == class1) & (y_pred_decoded == class2))
    TN = np.sum((y_decoded == class2) & (y_pred_decoded == class2))

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    return accuracy, TP, FP, FN, TN

def activation_fn(x, activation_function):
    if activation_function == "tanh":
        positive_expo = np.exp(x)
        negative_expo = np.exp(-x)
        return (positive_expo - negative_expo) / (positive_expo + negative_expo)
    elif activation_function == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation_function == "linear":
        return x;
    elif activation_function == "signum":
        return 1 if x > 0 else -1 if x < 0 else 0
    else:
        print("Invalid activation function")

def activation_fn_derivative(z, activation_function):
    if activation_function == "tanh":
        return 1 - z**2
    elif activation_function == "sigmoid":
        return z * (1-z)

def preprocessing(algorithm, feature1=None, feature2=None, class1=None, class2=None):
    if algorithm == "slp" or algorithm == "Adaline":
        df = pd.read_csv('birds_data.csv')
        df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
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
        df = pd.read_csv('birds_data.csv')
        
        # Handle missing values
        df["gender"].fillna(df["gender"].mode()[0], inplace=True)
        
        # Select numeric features
        numeric_features = ['beak_length', 'beak_depth', 'body_mass', 'fin_length']
        X_numeric = df[numeric_features].values
        
        # One-hot encode gender using scikit-learn
        gender_encoder = OneHotEncoder(sparse_output=False)
        X_categorical = gender_encoder.fit_transform(df[['gender']])
        
        # Normalize numeric features
        scaler = StandardScaler()
        X_numeric_normalized = scaler.fit_transform(X_numeric)
        
        # Combine normalized numeric features with categorical features
        X = np.hstack([X_numeric_normalized, X_categorical])
        
        # One-hot encode the target variable
        target_encoder = OneHotEncoder(sparse_output=False)
        y = target_encoder.fit_transform(df[['bird category']])
        
        # Get unique classes
        unique_classes = df['bird category'].unique()
        
        # Initialize empty arrays for train and test sets
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        
        # For each class, select 30 samples for training and 20 for testing
        for class_name in unique_classes:
            # Get indices for current class
            class_indices = df[df['bird category'] == class_name].index
            
            # Shuffle the indices
            np.random.seed(42)  # For reproducibility
            shuffled_indices = np.random.permutation(class_indices)
            
            # Split into train and test
            train_indices = shuffled_indices[:30]
            test_indices = shuffled_indices[30:50]  # Take next 20 samples
            
            # Add to train and test sets
            X_train_list.append(X[train_indices])
            X_test_list.append(X[test_indices])
            y_train_list.append(y[train_indices])
            y_test_list.append(y[test_indices])
        
        # Combine all classes
        X_train = np.vstack(X_train_list)
        X_test = np.vstack(X_test_list)
        y_train = np.vstack(y_train_list)
        y_test = np.vstack(y_test_list)
        
        # Final shuffle of the combined data
        train_indices = np.random.permutation(len(X_train))
        test_indices = np.random.permutation(len(X_test))
        
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        
        return X_train, y_train, X_test, y_test
    else:
        print("Invalid Algorithm")