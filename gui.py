import tkinter as tk
from tkinter import ttk, messagebox
import slp
import adaline

def toggle_bias_entry():
    if bias_var.get():
        bias_label.pack()
        bias_entry.pack()
    else:
        bias_label.pack_forget()
        bias_entry.pack_forget()

def toggle_mse_entry():
    if algo_var.get() == "Adaline":
        mse_label.pack()
        mse_entry.pack()
    else:
        mse_label.pack_forget()
        mse_entry.pack_forget()

def train_model():
    try:
        feature1 = feature1_var.get()
        feature2 = feature2_var.get()
        class1 = class1_var.get()
        class2 = class2_var.get()
        eta = float(eta_entry.get())
        epochs = int(epochs_entry.get())
        mse_threshold = float(mse_entry.get()) if algo_var.get() == "Adaline" else "N/A"
        bias = bias_var.get()
        bias_value = float(bias_entry.get()) if bias else "N/A"
        algorithm = algo_var.get()
        
        if not feature1 or not feature2 or not class1 or not class2:
            messagebox.showerror("Input Error", "Please select both features and classes.")
            return
        
        if algorithm == "Perceptron":
            slp.main(feature1, feature2, class1, class2, eta, epochs, bias, bias_value)
        else:
            adaline.main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias, bias_value)
        
        result_label.config(text=f"Training {algorithm} with:\nFeatures: {feature1}, {feature2}\nClasses: {class1}, {class2}\nLearning Rate: {eta}\nEpochs: {epochs}\nMSE Threshold: {mse_threshold}\nBias: {'Yes' if bias else 'No'}\nBias Value: {bias_value}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for Learning Rate, Number of Epochs, MSE Threshold, and Bias Value.")

# Main window
root = tk.Tk()
root.title("Perceptron & Adaline Trainer")
root.geometry("400x600")

# Feature selection
feature_options = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
feature1_var = tk.StringVar()
feature2_var = tk.StringVar()

ttk.Label(root, text="Select Feature 1:").pack()
ttk.OptionMenu(root, feature1_var, *feature_options).pack()

ttk.Label(root, text="Select Feature 2:").pack()
ttk.OptionMenu(root, feature2_var, *feature_options).pack()

# Class selection
class_options = ["Class 1", "Class 2", "Class 3"]
class1_var = tk.StringVar()
class2_var = tk.StringVar()

ttk.Label(root, text="Select Class 1:").pack()
ttk.OptionMenu(root, class1_var, *class_options).pack()

ttk.Label(root, text="Select Class 2:").pack()
ttk.OptionMenu(root, class2_var, *class_options).pack()

# Learning rate
ttk.Label(root, text="Learning Rate (eta):").pack()
eta_entry = ttk.Entry(root)
eta_entry.pack()

# Epochs
ttk.Label(root, text="Number of Epochs:").pack()
epochs_entry = ttk.Entry(root)
epochs_entry.pack()

# MSE Threshold (Initially Hidden)
mse_label = ttk.Label(root, text="MSE Threshold:")
mse_entry = ttk.Entry(root)

# Algorithm Selection
algo_var = tk.StringVar(value="Perceptron")
ttk.Label(root, text="Select Algorithm:").pack()
ttk.Radiobutton(root, text="Perceptron", variable=algo_var, value="Perceptron", command=toggle_mse_entry).pack()
ttk.Radiobutton(root, text="Adaline", variable=algo_var, value="Adaline", command=toggle_mse_entry).pack()

# Bias Checkbox
bias_var = tk.BooleanVar()
ttk.Checkbutton(root, text="Add Bias", variable=bias_var, command=toggle_bias_entry).pack()

# Bias Input (Initially Hidden)
bias_label = ttk.Label(root, text="Bias Value:")
bias_entry = ttk.Entry(root)

# Train Button
ttk.Button(root, text="Train", command=train_model).pack()

# Result Label
result_label = ttk.Label(root, text="")
result_label.pack()

# Update dropdown backgrounds to white
style = ttk.Style()
style.configure("TMenubutton", background="white")

root.mainloop()
