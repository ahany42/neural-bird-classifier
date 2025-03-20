import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('birds_data.csv')

numerical_features = ['beak_length', 'beak_depth', 'body_mass', 'fin_length']

# Function to display gender distribution pie chart
def gender_distribution():
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['blue', 'pink'])
        plt.title("Gender Distribution")
        plt.show()
    else:
        messagebox.showerror("Error", "'gender' column not found in the dataset.")

# Function to display correlation matrix
def correlation_matrix():
    plt.figure(figsize=(8, 6))
    correlation = df[numerical_features].corr()
    plt.matshow(correlation, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(numerical_features)), numerical_features, rotation=90)
    plt.yticks(range(len(numerical_features)), numerical_features)
    plt.title("Correlation Matrix")
    plt.show()

# Function to display feature distributions as subplots
def feature_distributions():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid for 4 features
    axs = axs.ravel()  # Flatten the 2x2 array into 1D for easy iteration

    for i, feature in enumerate(numerical_features):
        axs[i].hist(df[feature], bins=20, alpha=0.7, label=feature)
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel("Frequency")
        axs[i].set_title(f"Distribution of {feature}")
        axs[i].legend(loc='best')

    plt.tight_layout()
    plt.show()

# Function to display scatter plot matrix
def scatter_plot_matrix():
    pd.plotting.scatter_matrix(df[numerical_features], figsize=(10, 10), alpha=0.7)
    plt.show()

# Function to display box plots as subplots
def box_plots():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid for 4 features
    axs = axs.ravel()  # Flatten the 2x2 array into 1D for easy iteration

    for i, feature in enumerate(numerical_features):
        axs[i].boxplot(df[feature])
        axs[i].set_title(f"Boxplot of {feature}")
        axs[i].set_ylabel('Values')
        axs[i].set_xticklabels([feature])  # Set the x-axis label for each plot

    plt.tight_layout()
    plt.show()

# Function to display scatter plots as subplots
def scatter_plots():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid for 4 scatter plots
    axs = axs.ravel()  # Flatten the 2x2 array into 1D for easy iteration

    for i, feature in enumerate(numerical_features):
        axs[i].scatter(df[feature], df['beak_length'])  # Example: scatter feature vs beak_length
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('beak_length')
        axs[i].set_title(f"Scatter {feature} vs beak_length")

    plt.tight_layout()
    plt.show()

# Create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Data Analysis")

    # Set window size to be bigger
    root.geometry("400x400")  # Adjust the size as per your preference

    # Gender Distribution Button
    gender_button = tk.Button(root, text="Gender Distribution Pie Chart", command=gender_distribution)
    gender_button.pack(pady=10)

    # Correlation Matrix Button
    correlation_button = tk.Button(root, text="Correlation Matrix", command=correlation_matrix)
    correlation_button.pack(pady=10)

    # Feature Distributions Button
    feature_button = tk.Button(root, text="Feature Distributions", command=feature_distributions)
    feature_button.pack(pady=10)

    # Scatter Plot Matrix Button
    scatter_button = tk.Button(root, text="Scatter Plot Matrix", command=scatter_plot_matrix)
    scatter_button.pack(pady=10)

    # Box Plots Button
    box_button = tk.Button(root, text="Box Plots", command=box_plots)
    box_button.pack(pady=10)

    # Scatter Plots Button
    scatter_plots_button = tk.Button(root, text="Scatter Plots", command=scatter_plots)
    scatter_plots_button.pack(pady=10)

    root.mainloop()

#
