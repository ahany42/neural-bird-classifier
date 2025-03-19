import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df=pd.read_csv('birds_data.csv')
df['gender'] = df['gender'].fillna('unknown')
df = pd.get_dummies(df, columns=['gender'], prefix='gender')

print(df[df['gender_unknown'] == 1])
print(df.head(30))

""""
print(df.describe())

print(df.isnull().sum())
print(f"Total Duplicates: {df.duplicated().sum()}")

# Z-score outlier
z_scores = stats.zscore(df.select_dtypes(include='number'))
outliers = (abs(z_scores) > 3).any(axis=1)
print("Outliers (Z-score > 3):")
print(df[outliers])

# IQR 
Q1 = df['beak_length'].quantile(0.25)
Q3 = df['beak_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['beak_length'] < lower_bound) | (df['beak_length'] > upper_bound)]
print("Outliers (IQR Method):")
print(outliers_iqr)
"""
