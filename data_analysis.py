import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "D:/birds.csv"
df = pd.read_csv(file_path)

#features that r analyzed
features = ['beak_length', 'beak_depth', 'body_mass', 'fin_length']

def display_menu():
    print("Select an analysis to perform:")
    print("1 - Gender Distribution Pie Chart")
    print("2 - Correlation Matrix")
    print("3 - Feature Distributions")
    print("4 - Scatter Plot Matrix")
    print("5 - Box Plots")
    print("0 - Exit")

while True:
    display_menu()
    choice = input("Enter your choice: ")

    if choice == "1":
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['blue', 'pink'])
            plt.title("Gender Distribution")
            plt.show()

    elif choice == "2":
        print("\nCorrelation Matrix:")
        print(df[features].corr())

    elif choice == "3":
        for feature in features: #loop for each featre
            plt.hist(df[feature], bins=20, alpha=0.7, label=feature)
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {feature}")
            plt.show()

    elif choice == "4":
        pd.plotting.scatter_matrix(df[features], figsize=(10, 10), alpha=0.7)
        plt.show()

    elif choice == "5":
        for feature in features:
            plt.boxplot(df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()

    elif choice == "0":
        break

    else:
        print("Invalid choice, please try again.")
