import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('birds_data.csv')

numerical_features = ['beak_length', 'beak_depth', 'body_mass', 'fin_length']
def apply_pca():
    X = df[numerical_features].fillna(df[numerical_features].mean())  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', alpha=0.7)
    plt.title("PCA of Numerical Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    feature_loading = pd.DataFrame(pca.components_, columns=numerical_features, index=["PC1", "PC2"])
    plt.figure(figsize=(8, 6))
    plt.barh(numerical_features, feature_loading.loc["PC1"], label="PC1", alpha=0.7)
    plt.barh(numerical_features, feature_loading.loc["PC2"], label="PC2", alpha=0.7)
    plt.xlabel("Feature Importance")
    plt.title("Feature Contribution to Principal Components")
    plt.legend()
    plt.tight_layout()
    plt.show()

def gender_distribution():
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['blue', 'pink'])
        plt.title("Gender Distribution")
        plt.show()
    else:
        messagebox.showerror("Error", "'gender' column not found in the dataset.")

def correlation_matrix():
    plt.figure(figsize=(8, 6))
    correlation = df[numerical_features].corr()
    plt.matshow(correlation, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(numerical_features)), numerical_features, rotation=90)
    plt.yticks(range(len(numerical_features)), numerical_features)
    plt.title("Correlation Matrix")
    plt.show()

def feature_distributions():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  
    axs = axs.ravel() 

    for i, feature in enumerate(numerical_features):
        axs[i].hist(df[feature], bins=20, alpha=0.7, label=feature)
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel("Frequency")
        axs[i].set_title(f"Distribution of {feature}")
        axs[i].legend(loc='best')

    plt.tight_layout()
    plt.show()

def scatter_plot_matrix():
    pd.plotting.scatter_matrix(df[numerical_features], figsize=(10, 10), alpha=0.7)
    plt.show()

def box_plots():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  
    axs = axs.ravel() 

    for i, feature in enumerate(numerical_features):
        axs[i].boxplot(df[feature])
        axs[i].set_title(f"Boxplot of {feature}")
        axs[i].set_ylabel('Values')
        axs[i].set_xticklabels([feature])  

    plt.tight_layout()
    plt.show()

def scatter_plots():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  
    axs = axs.ravel()  

    for i, feature in enumerate(numerical_features):
        axs[i].scatter(df[feature], df['beak_length'])  
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('beak_length')
        axs[i].set_title(f"Scatter {feature} vs beak_length")

    plt.tight_layout()
    plt.show()

def create_gui():
    root = tk.Tk()
    root.title("Data Analysis")

    root.geometry("400x400")  
    
    gender_button = tk.Button(root, text="Gender Distribution Pie Chart", command=gender_distribution)
    gender_button.pack(pady=10)

    correlation_button = tk.Button(root, text="Correlation Matrix", command=correlation_matrix)
    correlation_button.pack(pady=10)

    feature_button = tk.Button(root, text="Feature Distributions", command=feature_distributions)
    feature_button.pack(pady=10)

    scatter_button = tk.Button(root, text="Scatter Plot Matrix", command=scatter_plot_matrix)
    scatter_button.pack(pady=10)

    box_button = tk.Button(root, text="Box Plots", command=box_plots)
    box_button.pack(pady=10)

    scatter_plots_button = tk.Button(root, text="Scatter Plots", command=scatter_plots)
    scatter_plots_button.pack(pady=10)

    pca_button = tk.Button(root, text="Apply PCA", command=apply_pca)
    pca_button.pack(pady=10)

    root.mainloop()


