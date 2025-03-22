import tkinter as tk
from tkinter import ttk, messagebox
import slp
import adaline
import mlp
import pandas as pd
import data_analysis

df = pd.read_csv('birds_data.csv')

DEFAULT_FEATURE = "Select Feature"
DEFAULT_CLASS = "Select Class"

def toggle_mse_entry():
    """Toggles the visibility of MSE and hidden layer fields based on the selected algorithm."""
    if algo_var.get() == "Adaline":
        mse_label.grid(row=9, column=0, sticky='w', pady=(5, 0))
        mse_entry.grid(row=9, column=1, pady=(5, 10))
        hidden_layers_label.grid_forget()
        hidden_layer_entry.grid_forget()
        activation_label.grid_forget()
        sigmoid_radio.grid_forget()
        tanh_radio.grid_forget()
    elif algo_var.get() == "mlp":
        mse_label.grid(row=9, column=0, sticky='w', pady=(5, 0))
        mse_entry.grid(row=9, column=1, pady=(5, 10))
        hidden_layers_label.grid(row=10, column=0, sticky='w', pady=(5, 0))
        hidden_layer_entry.grid(row=10, column=1, pady=(5, 10))
        activation_label.grid(row=11, column=0, sticky='w', pady=(5, 0))
        sigmoid_radio.grid(row=11, column=1, sticky='w', pady=(5, 0))
        tanh_radio.grid(row=11, column=2, sticky='w', pady=(5, 0))
    else:
        mse_label.grid_forget()
        mse_entry.grid_forget()
        hidden_layers_label.grid_forget()
        hidden_layer_entry.grid_forget()
        activation_label.grid_forget()
        sigmoid_radio.grid_forget()
        tanh_radio.grid_forget()

def train_model():
    """Trains the selected algorithm with user-provided parameters."""
    try:
        result_label.config(text="")  
        feature1, feature2 = feature1_var.get(), feature2_var.get()
        class1, class2 = class1_var.get(), class2_var.get()
        eta_text, epochs_text = eta_entry.get().strip(), epochs_entry.get().strip()
        mse_text = mse_entry.get().strip() if algo_var.get() == "Adaline" else "N/A"
        bias = bias_var.get()
        algorithm = algo_var.get()

        if feature1 == DEFAULT_FEATURE or feature2 == DEFAULT_FEATURE:
            messagebox.showerror("Input Error", "Please select both features.")
            return
        if feature1 == feature2:
            messagebox.showerror("Input Error", "Selected features must be different.")
            return
        if class1 == DEFAULT_CLASS or class2 == DEFAULT_CLASS:
            messagebox.showerror("Input Error", "Please select both classes.")
            return
        if class1 == class2:
            messagebox.showerror("Input Error", "Selected classes must be different.")
            return
        
        try:
            eta = float(eta_text)
            epochs = int(epochs_text)
            mse_threshold = float(mse_text) if algo_var.get() == "Adaline" else "N/A"
            hidden_layers = int(hidden_layer_entry.get()) if algo_var.get() == "mlp" else "N/A"
        except ValueError:
            if algorithm == "slp":
                messagebox.showerror("Input Error", "Ensure Learning Rate and Epochs are valid numbers.")
            elif algorithm == "Adaline":
                messagebox.showerror("Input Error", "Ensure Learning Rate, Epochs, and MSE Threshold are valid numbers.")
            elif algorithm == "mlp":
                messagebox.showerror("Input Error", "Ensure Learning Rate, Epochs, Number of neurons and Hidden Layers are valid numbers.")
            return

        if algorithm == "slp":
            accuracy, TP, FP, FN, TN = slp.main(feature1, feature2, class1, class2, eta, epochs, bias)
        elif algorithm == "Adaline":
            accuracy, TP, FP, FN, TN = adaline.main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias)
        elif algorithm == "mlp":
            activation_function = activation_var.get()
            accuracy, TP, FP, FN, TN = mlp.main(eta, epochs, bias, hidden_layers, activation_function)

        result_label.config(text=f"""
        Confusion Matrix:
        Predicted |  {class1}  |  {class2}  |
        ----------------------
        Actual {class1} | {TP:3} | {FN:3} |
        Actual {class2} | {FP:3} | {TN:3} |

        Accuracy: {accuracy:.2f} %
        """)
        result_label.grid(row=13, column=0, columnspan=3, pady=10)
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

def open_data_analysis():
    """Opens the data analysis module."""
    try:
        data_analysis.create_gui()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run data analysis: {e}")

root = tk.Tk()
root.title("SLP, Adaline & MLP Trainer")
root.geometry("450x750")
root.configure(padx=50, pady=20)

feature_options = df.columns[:-1].tolist()
class_options = df['bird category'].unique()

frame = ttk.Frame(root)
frame.pack(expand=True)

feature1_var = tk.StringVar(value=DEFAULT_FEATURE)
feature2_var = tk.StringVar(value=DEFAULT_FEATURE)
class1_var = tk.StringVar(value=DEFAULT_CLASS)
class2_var = tk.StringVar(value=DEFAULT_CLASS)
algo_var = tk.StringVar(value="slp")
activation_var = tk.StringVar(value="sigmoid")
bias_var = tk.BooleanVar()

ttk.Label(frame, text="Select Features:").grid(row=0, column=0, columnspan=2, sticky='w', pady=(5, 0))
ttk.OptionMenu(frame, feature1_var, DEFAULT_FEATURE, *feature_options).grid(row=1, column=0, padx=5, pady=(0, 10))
ttk.OptionMenu(frame, feature2_var, DEFAULT_FEATURE, *feature_options).grid(row=1, column=1, padx=5, pady=(0, 10))

ttk.Label(frame, text="Select Classes:").grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))
ttk.OptionMenu(frame, class1_var, DEFAULT_CLASS, *class_options).grid(row=3, column=0, padx=5, pady=(0, 10))
ttk.OptionMenu(frame, class2_var, DEFAULT_CLASS, *class_options).grid(row=3, column=1, padx=5, pady=(0, 10))

ttk.Label(frame, text="Learning Rate (eta):").grid(row=4, column=0, sticky='w', pady=(5, 0))
eta_entry = ttk.Entry(frame)
eta_entry.grid(row=4, column=1, pady=(0, 10))
ttk.Label(frame, text="Number of Epochs:").grid(row=5, column=0, sticky='w', pady=(5, 0))
epochs_entry = ttk.Entry(frame)
epochs_entry.grid(row=5, column=1, pady=(0, 10))

mse_label = ttk.Label(frame, text="MSE Threshold:")
mse_entry = ttk.Entry(frame)
hidden_layers_label = ttk.Label(frame, text="Hidden Layers:")
hidden_layer_entry = ttk.Entry(frame)
activation_label = ttk.Label(frame, text="Activation Function:")
sigmoid_radio = ttk.Radiobutton(frame, text="Sigmoid", variable=activation_var, value="sigmoid")
tanh_radio = ttk.Radiobutton(frame, text="Tanh", variable=activation_var, value="tanh")

ttk.Label(frame, text="Select Algorithm:").grid(row=6, column=0, columnspan=2, sticky='w', pady=(5, 0))
for i, name in enumerate(["slp", "Adaline", "mlp"]):
    ttk.Radiobutton(frame, text=name, variable=algo_var, value=name, command=toggle_mse_entry).grid(row=8, column=i, sticky='w')

ttk.Button(frame, text="Train", command=train_model).grid(row=16, column=0, columnspan=2, pady=(10, 10))
ttk.Button(frame, text="Data Analysis", command=open_data_analysis).grid(row=17, column=0, columnspan=2, pady=(10, 10))

result_label = ttk.Label(frame, text="")
result_label.grid(row=18, column=0, columnspan=2, pady=(10, 10))
style = ttk.Style()
style.configure("TMenubutton", background="white")
root.geometry("500x700")
root.mainloop()
