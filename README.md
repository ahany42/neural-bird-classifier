# 🦜 Bird Classification using Perceptron (SLP) & ADALINE

A single-layer neural network implementation to classify bird species using the **Perceptron (SLP)** and **ADALINE** algorithms. The project includes **data analysis**, **training & testing**, **implementing Mean Squared Error (MSE)**, **activation functions**, and a **GUI for visualization**.

---

## 🚀 Features

- ✅ Implementation of **Single Layer Perceptron (SLP)**
- ✅ Implementation of **ADALINE** with **MSE loss**
- ✅ Custom **training & testing** (30/20 split per class)
- ✅ **Data analysis** (feature distribution, correlation, and visualizations)
- ✅ **Graphical Interface (GUI)** for user-friendly interaction
- ✅ Implementation of **activation functions** for model training

---

## 📦 Imports & Dependencies

This project requires the following Python libraries:

- `numpy` - For numerical operations
- `pandas` - For data manipulation and analysis
- `matplotlib` - For data visualization
- `seaborn` - For enhanced visualizations

You can install them using:

```bash
pip install numpy pandas matplotlib seaborn
```

---

## 📁 Project Structure

```bash
/bird-classification
│── main.py          # Main entry point to train and test models
│── slp.py           # Perceptron Learning Algorithm (SLP)
│── adaline.py       # ADALINE Learning Algorithm (MSE)
│── utils.py         # Common functions (data loading, splitting, visualization)
│── gui.py           # GUI for visualization & interaction
│── data_analysis.py # Exploratory Data Analysis (EDA) & graphs
│── birds_data.csv   # Dataset (bird features & classes)
│── README.md        # Project documentation
```

## 📊 Data Analysis & Visualization

- **📈 Feature Distribution Histograms**: Visualizes feature distribution and frequency.
- **📊 Class Distribution Plots**: Shows class balance in dataset.
- **🔥 Correlation Matrix Heatmap**: Displays feature pair correlations.
- **🧩 Decision Boundary Visualization**: Visualizes model decision boundaries.
- **🥧 Gender Distribution Pie Chart**: Shows gender proportions in dataset.
- **📊 Feature Distributions**: Displays feature value distribution.
- **🔗 Scatter Plot Matrix**: Shows pairwise feature relationships.
- **📦 Box Plots**: Displays distribution, spread, and outliers.
- **🔍 PCA (Principal Component Analysis)**: Reduces dimensions, preserves variance.
- **🔄 Pair Plot**: Grid of feature pair scatter plots.

---

## 🧠 Algorithms Implemented

### **1️⃣ Single-Layer Perceptron (SLP)**

- Uses **signum activation function**
- Trained with **perceptron learning rule**
- Outputs **binary class labels**

### **2️⃣ ADALINE (Adaptive Linear Neuron)**

- Uses **linear activation function during training**
- Uses **signum activation function during testing**
- Optimized using **Mean Squared Error (MSE)**
- Outputs **continuous values** before thresholding

## 📊 Results & Performance

| Model      | Accuracy (%) |
| ---------- | ------------ |
| Perceptron | up to 100%   |
| ADALINE    | up to 100%   |

### Best Features for ADALINE

- **Best Combination:** Body mass and Beak Depth for categories a and B.

### Best Features for Perceptron (SLP)

- **Best Combination:** Beak depth and beak length for categories A and C are perfectly separable.

