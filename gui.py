import tkinter as tk
from tkinter import ttk, messagebox
import slp
import adaline
import pandas as pd
import data_analysis
df = pd.read_csv('birds_data.csv')

DEFAULT_FEATURE = "Select Feature"
DEFAULT_CLASS = "Select Class"

def start_gui():
    root.mainloop()

def toggle_mse_entry():
    if algo_var.get() == "Adaline":
        mse_label.grid(row=9, column=0, sticky='w', pady=(5, 0))
        mse_entry.grid(row=9, column=1, pady=(5, 15))
    else:
        mse_label.grid_forget()
        mse_entry.grid_forget()

def train_model():
    try:
        result_label.config(text="")
        feature1 = feature1_var.get()
        feature2 = feature2_var.get()
        class1 = class1_var.get()
        class2 = class2_var.get()
        eta_text = eta_entry.get().strip()
        epochs_text = epochs_entry.get().strip()
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
        except ValueError:
            messagebox.showerror("Input Error", "Ensure Learning Rate, Epochs, and MSE Threshold (if applicable) are valid numbers.")
            return
        
        if algorithm == "slp":
            accuracy ,TP,FP,FN,TN = slp.main(feature1, feature2, class1, class2, eta, epochs, bias)
        else:
            accuracy ,TP,FP,FN,TN = adaline.main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias)

        result_label.config(text=f"""
        Confusion Matrix:
        Predicted |  {class1}  |  {class2}  |
        ----------------------
        Actual {class1} | {TP:3} | {FN:3} |
        Actual {class2} | {FP:3} | {TN:3} |

        Accuracy: {accuracy} %
        """)
        result_label.grid(row=10, column=0, columnspan=2, pady=10)  
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

root = tk.Tk()
root.title("slp & Adaline Trainer")
root.geometry("400x600")
root.configure(padx=50, pady=20)

feature_options = df.columns[:-1].tolist()
class_options = df['bird category'].unique()

frame = ttk.Frame(root)
frame.pack(expand=True)

feature1_var = tk.StringVar(value=DEFAULT_FEATURE)
feature2_var = tk.StringVar(value=DEFAULT_FEATURE)
class1_var = tk.StringVar(value=DEFAULT_CLASS)
class2_var = tk.StringVar(value=DEFAULT_CLASS)

ttk.Label(frame, text="Select Features:").grid(row=0, column=0, columnspan=2, sticky='w', pady=(5, 0))
feature1_menu = ttk.OptionMenu(frame, feature1_var, DEFAULT_FEATURE, *feature_options)
feature1_menu.grid(row=1, column=0, padx=5, pady=(0, 10))

feature2_menu = ttk.OptionMenu(frame, feature2_var, DEFAULT_FEATURE, *feature_options)
feature2_menu.grid(row=1, column=1, padx=5, pady=(0, 10))

ttk.Label(frame, text="Select Classes:").grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))
class1_menu = ttk.OptionMenu(frame, class1_var, DEFAULT_CLASS, *class_options)
class1_menu.grid(row=3, column=0, padx=5, pady=(0, 10))

class2_menu = ttk.OptionMenu(frame, class2_var, DEFAULT_CLASS, *class_options)
class2_menu.grid(row=3, column=1, padx=5, pady=(0, 10))

ttk.Label(frame, text="Learning Rate (eta):").grid(row=4, column=0, sticky='w', pady=(5, 0))
eta_entry = ttk.Entry(frame)
eta_entry.grid(row=4, column=1, pady=(0, 10))

ttk.Label(frame, text="Number of Epochs:").grid(row=5, column=0, sticky='w', pady=(5, 0))
epochs_entry = ttk.Entry(frame)
epochs_entry.grid(row=5, column=1, pady=(0, 10))

mse_label = ttk.Label(frame, text="MSE Threshold:")
mse_entry = ttk.Entry(frame)

algo_var = tk.StringVar(value="slp")
ttk.Label(frame, text="Select Algorithm:").grid(row=6, column=0, columnspan=2, sticky='w', pady=(5, 0))
ttk.Radiobutton(frame, text="slp", variable=algo_var, value="slp", command=toggle_mse_entry).grid(row=8, column=0, sticky='w', pady=(0, 10))
ttk.Radiobutton(frame, text="Adaline", variable=algo_var, value="Adaline", command=toggle_mse_entry).grid(row=8, column=1, sticky='w', pady=(0, 10))

bias_var = tk.BooleanVar()
ttk.Checkbutton(frame, text="Add Bias", variable=bias_var).grid(row=12, column=0, columnspan=2, sticky='w', pady=(5, 0))

ttk.Button(frame, text="Train", command=train_model).grid(row=18, column=0, columnspan=2, pady=(10, 10))
ttk.Button(frame, text="Data Analysis", command=data_analysis.create_gui).grid(row=18, column=0, columnspan=2, pady=(100, 10))
result_label = ttk.Label(frame, text="")
result_label.grid(row=12, column=0, columnspan=2, pady=(10, 10))

style = ttk.Style()
style.configure("TMenubutton", background="white")

root.mainloop()
