import tkinter as tk
from tkinter import ttk, messagebox
import slp
import adaline
import pandas as pd

df = pd.read_csv('birds_data.csv')
def start_gui():
    root.mainloop()

if __name__ == "__main__":
    start_gui()
    
def toggle_mse_entry():
    if algo_var.get() == "Adaline":
        mse_label.grid(row=9, column=0, sticky='w', pady=(5, 0))
        mse_entry.grid(row=9, column=1, pady=(5, 15))
    else:
        mse_label.grid_forget()
        mse_entry.grid_forget()

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
        algorithm = algo_var.get()

        if not feature1 or not feature2 or not class1 or not class2:
            messagebox.showerror("Input Error", "Please select both features and classes.")
            return

        elif feature1 == feature2:
            messagebox.showerror("Input Error", "Selected features must be different.")
            return

        elif class1 == class2:
            messagebox.showerror("Input Error", "Selected classes must be different.")
            return

        elif algorithm == "Perceptron":
            slp.main(feature1, feature2, class1, class2, eta, epochs, bias)
        else:
            adaline.main(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias)

        result_label.config(text=f"Training {algorithm} with:\n"
                                 f"Features: {feature1}, {feature2}\n"
                                 f"Classes: {class1}, {class2}\n"
                                 f"Learning Rate: {eta}\n"
                                 f"Epochs: {epochs}\n"
                                 f"MSE Threshold: {mse_threshold}\n"
                                 f"Bias: {'Yes' if bias else 'No'}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for Learning Rate, Number of Epochs, and MSE Threshold.")

root = tk.Tk()
root.title("Perceptron & Adaline Trainer")
root.geometry("400x600")
root.configure(padx=50, pady=20)

feature_options = df.columns[:-1].tolist()
class_options = df['bird category'].unique()

frame = ttk.Frame(root)
frame.pack(expand=True)

feature1_var = tk.StringVar()
feature2_var = tk.StringVar()
class1_var = tk.StringVar()
class2_var = tk.StringVar()

ttk.Label(frame, text="Select Features:").grid(row=0, column=0, columnspan=2, sticky='w', pady=(5, 0))
ttk.OptionMenu(frame, feature1_var, *feature_options).grid(row=1, column=0, padx=5, pady=(0, 10))
ttk.OptionMenu(frame, feature2_var, *feature_options).grid(row=1, column=1, padx=5, pady=(0, 10))


ttk.Label(frame, text="Select Classes:").grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))
ttk.OptionMenu(frame, class1_var, *class_options).grid(row=3, column=0, padx=5, pady=(0, 10))
ttk.OptionMenu(frame, class2_var, *class_options).grid(row=3, column=1, padx=5, pady=(0, 10))


ttk.Label(frame, text="Learning Rate (eta):").grid(row=4, column=0, sticky='w', pady=(5, 0))
eta_entry = ttk.Entry(frame)
eta_entry.grid(row=4, column=1, pady=(0, 10))


ttk.Label(frame, text="Number of Epochs:").grid(row=5, column=0, sticky='w', pady=(5, 0))
epochs_entry = ttk.Entry(frame)
epochs_entry.grid(row=5, column=1, pady=(0, 10))


mse_label = ttk.Label(frame, text="MSE Threshold:")
mse_entry = ttk.Entry(frame)


algo_var = tk.StringVar(value="Perceptron")
ttk.Label(frame, text="Select Algorithm:").grid(row=6, column=0, columnspan=2, sticky='w', pady=(5, 0))
ttk.Radiobutton(frame, text="Perceptron", variable=algo_var, value="Perceptron", command=toggle_mse_entry).grid(row=8, column=0, sticky='w', pady=(0, 10))
ttk.Radiobutton(frame, text="Adaline", variable=algo_var, value="Adaline", command=toggle_mse_entry).grid(row=8, column=1, sticky='w', pady=(0, 10))


bias_var = tk.BooleanVar()
ttk.Checkbutton(frame, text="Add Bias", variable=bias_var).grid(row=12, column=0, columnspan=2, sticky='w', pady=(5, 0))


ttk.Button(frame, text="Train", command=train_model).grid(row=18, column=0, columnspan=2, pady=(10, 10))


result_label = ttk.Label(frame, text="")
result_label.grid(row=12, column=0, columnspan=2, pady=(10, 10))

style = ttk.Style()
style.configure("TMenubutton", background="white")

root.mainloop()
