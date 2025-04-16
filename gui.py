import tkinter as tk
from tkinter import ttk, messagebox
import slp
import adaline
import mlp
import pandas as pd
import data_analysis
import numpy as np

df = pd.read_csv('birds_data.csv')

DEFAULT_FEATURE = "Select Feature"
DEFAULT_CLASS = "Select Class"
def start_gui():
    root.mainloop()

def toggle_mse_entry():
    if algo_var.get() == "Adaline":
        mse_label.grid(row=9, column=0, sticky='w', pady=(5, 0))
        mse_entry.grid(row=9, column=1, pady=(5, 10))
        hidden_layers_label.grid_forget()
        hidden_layer_entry.grid_forget()
        neurons_label.grid_forget()
        neurons_entry.grid_forget()
        activation_label.grid_forget()
        sigmoid_radio.grid_forget()
        tanh_radio.grid_forget()
        bias_checkbox.grid(row=7, column=0, sticky='w', pady=(5, 0))
    elif algo_var.get() == "mlp":
        mse_label.grid(row=9, column=0, sticky='w', pady=(5, 0))
        mse_entry.grid(row=9, column=1, pady=(5, 10))
        hidden_layers_label.grid(row=10, column=0, sticky='w', pady=(5, 0))
        hidden_layer_entry.grid(row=10, column=1, pady=(5, 10))
        neurons_label.grid(row=11, column=0, sticky='w', pady=(5, 0))
        neurons_entry.grid(row=11, column=1, pady=(5, 10))
        activation_label.grid(row=12, column=0, sticky='w', pady=(5, 0))
        sigmoid_radio.grid(row=12, column=1, sticky='w', pady=(5, 0))
        tanh_radio.grid(row=12, column=2, sticky='w', pady=(5, 0))
        bias_checkbox.grid(row=7, column=0, sticky='w', pady=(5, 0))
    else:
        mse_label.grid_forget()
        mse_entry.grid_forget()
        hidden_layers_label.grid_forget()
        hidden_layer_entry.grid_forget()
        neurons_label.grid_forget()
        neurons_entry.grid_forget()
        activation_label.grid_forget()
        sigmoid_radio.grid_forget()
        tanh_radio.grid_forget()
        bias_checkbox.grid(row=7, column=0, sticky='w', pady=(5, 0))

def format_confusion_matrix(confusion_matrix, classes):
    # Create header
    header = "Predicted | " + " | ".join(f"{c:^10}" for c in classes) + " |"
    separator = "-" * (len(header) + 2)
    
    # Create rows
    rows = []
    for i, actual in enumerate(classes):
        row = f"Actual {actual} | " + " | ".join(f"{confusion_matrix[i][j]:^10}" for j in range(len(classes))) + " |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)

def train_model():
    try:
        result_label.config(text="") 
        feature1, feature2 = feature1_var.get(), feature2_var.get()
        class1, class2 = class1_var.get(), class2_var.get()
        eta_text, epochs_text = eta_entry.get().strip(), epochs_entry.get().strip()
        mse_text = mse_entry.get().strip() if algo_var.get() == "Adaline" or algo_var.get() == "mlp" else "N/A"
        bias = bias_var.get()  
        algorithm = algo_var.get()

        if algorithm != "mlp":
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
            mse_threshold = float(mse_text) if algo_var.get() == "Adaline" or algo_var.get() == "mlp" else "N/A"
            hidden_layers = int(hidden_layer_entry.get()) if algo_var.get() == "mlp" else "N/A"
            neurons_per_layer = neurons_entry.get() if algo_var.get() == "mlp" else "N/A"
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
            result_text = f"""
            Confusion Matrix:
            Predicted |  {class1}  |  {class2}  |
            ----------------------
            Actual {class1} | {TP:3} | {FN:3} |
            Actual {class2} | {FP:3} | {TN:3} |

            Accuracy: {accuracy:.2f} %
            """
        elif algorithm == "Adaline":
            accuracy, TP, FP, FN, TN = adaline.main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias)
            result_text = f"""
            Confusion Matrix:
            Predicted |  {class1}  |  {class2}  |
            ----------------------
            Actual {class1} | {TP:3} | {FN:3} |
            Actual {class2} | {FP:3} | {TN:3} |

            Accuracy: {accuracy:.2f} %
            """
        elif algorithm == "mlp":
            activation_function = activation_var.get()
            confusion_matrix, accuracy = mlp.main(eta, epochs, bias, neurons_per_layer, hidden_layers, activation_function, mse_threshold)
            classes = df['bird category'].unique()
            result_text = f"""
            Confusion Matrix:
            {format_confusion_matrix(confusion_matrix, classes)}

            Overall Accuracy: {accuracy:.2f} %
            """

        result_label.config(text=result_text)
        result_label.grid(row=13, column=0, columnspan=3, pady=10)
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

def open_data_analysis():
    try:
        data_analysis.create_gui()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run data analysis: {e}")
        
def hide_feature_class_selection():
    if algo_var.get() == "mlp":
        feature1_label.grid_forget()
        feature1_dropdown.grid_forget()
        feature2_label.grid_forget()
        feature2_dropdown.grid_forget()
        class1_label.grid_forget()
        class1_dropdown.grid_forget()
        class2_label.grid_forget()
        class2_dropdown.grid_forget()
    else:
        feature1_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(5, 0))
        feature1_dropdown.grid(row=1, column=0, padx=5, pady=(0, 10))
        feature2_label.grid(row=0, column=1, columnspan=2, sticky='w', pady=(5, 0))
        feature2_dropdown.grid(row=1, column=1, padx=5, pady=(0, 10))
        class1_label.grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))
        class1_dropdown.grid(row=3, column=0, padx=5, pady=(0, 10))
        class2_label.grid(row=2, column=1, columnspan=2, sticky='w', pady=(5, 0))
        class2_dropdown.grid(row=3, column=1, padx=5, pady=(0, 10))

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
bias_var = tk.BooleanVar(value=False)  

feature1_label = ttk.Label(frame, text="Select Features:")
feature1_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(5, 0))
feature1_dropdown = ttk.OptionMenu(frame, feature1_var, DEFAULT_FEATURE, *feature_options)
feature1_dropdown.grid(row=1, column=0, padx=5, pady=(0, 10))

feature2_label = ttk.Label(frame, text="Select Features:")
feature2_label.grid(row=0, column=1, columnspan=2, sticky='w', pady=(5, 0))
feature2_dropdown = ttk.OptionMenu(frame, feature2_var, DEFAULT_FEATURE, *feature_options)
feature2_dropdown.grid(row=1, column=1, padx=5, pady=(0, 10))

class1_label = ttk.Label(frame, text="Select Classes:")
class1_label.grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))
class1_dropdown = ttk.OptionMenu(frame, class1_var, DEFAULT_CLASS, *class_options)
class1_dropdown.grid(row=3, column=0, padx=5, pady=(0, 10))

class2_label = ttk.Label(frame, text="Select Classes:")
class2_label.grid(row=2, column=1, columnspan=2, sticky='w', pady=(5, 0))
class2_dropdown = ttk.OptionMenu(frame, class2_var, DEFAULT_CLASS, *class_options)
class2_dropdown.grid(row=3, column=1, padx=5, pady=(0, 10))

ttk.Label(frame, text="Learning Rate (eta):").grid(row=4, column=0, sticky='w', pady=(5, 0))
eta_entry = ttk.Entry(frame)
eta_entry.grid(row=4, column=1, pady=(0, 10))
ttk.Label(frame, text="Number of Epochs:").grid(row=5, column=0, sticky='w', pady=(5, 0))
epochs_entry = ttk.Entry(frame)
epochs_entry.grid(row=5, column=1, pady=(0, 10))

bias_checkbox = ttk.Checkbutton(frame, text="Use Bias", variable=bias_var)
bias_checkbox.grid(row=7, column=0, sticky='w', pady=(5, 0))

mse_label = ttk.Label(frame, text="MSE Threshold:")
mse_entry = ttk.Entry(frame)
hidden_layers_label = ttk.Label(frame, text="Hidden Layers:")
hidden_layer_entry = ttk.Entry(frame)
neurons_label = ttk.Label(frame, text="Neurons per Layers Comma Separated:")
neurons_entry = ttk.Entry(frame)
activation_label = ttk.Label(frame, text="Activation Function:")
sigmoid_radio = ttk.Radiobutton(frame, text="Sigmoid", variable=activation_var, value="sigmoid")
tanh_radio = ttk.Radiobutton(frame, text="Tanh", variable=activation_var, value="tanh")

ttk.Label(frame, text="Select Algorithm:").grid(row=6, column=0, columnspan=2, sticky='w', pady=(5, 0))
for i, name in enumerate(["slp", "Adaline", "mlp"]):
    ttk.Radiobutton(frame, text=name, variable=algo_var, value=name, command=lambda: [toggle_mse_entry(), hide_feature_class_selection()]).grid(row=8, column=i, sticky='w')

ttk.Button(frame, text="Train", command=train_model).grid(row=16, column=0, columnspan=2, pady=(10, 10))
ttk.Button(frame, text="Data Analysis", command=open_data_analysis).grid(row=17, column=0, columnspan=2, pady=(10, 10))

result_label = ttk.Label(frame, text="")
result_label.grid(row=18, column=0, columnspan=2, pady=(10, 10))
style = ttk.Style()
style.configure("TMenubutton", background="white")
root.geometry("500x700")
