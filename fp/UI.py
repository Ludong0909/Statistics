import tkinter as tk
from tkinter import ttk, messagebox
import main_script as ms

def show_results():
    today_date = entry_date.get()
    try:
        today_temp = float(entry_temp.get())
        today_precip = float(entry_precip.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for temperature and precipitation.")
        return
    result = ms.compare_today_to_history(today_date, today_temp, today_precip, ms.station_names, ms.special_values)
    # result_text.set(f"Temperature Comparison: {result['temperature_comparison']}\nPrecipitation Comparison: {result['precipitation_comparison']}")
    result_text.set(f"Temperature Comparison: {result['temperature_comparison']}, {result['temperature_percentile']}\nPrecipitation Comparison: {result['precipitation_comparison']}, {result['precipitation_percentile']}")

def plot_pdf():
    today_date = entry_date.get()
    try:
        today_temp = float(entry_temp.get())
        today_precip = float(entry_precip.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for temperature and precipitation.")
        return
    ms.plot_historical_pdf_with_today(today_date, today_temp, today_precip, ms.station_names, ms.special_values)

# UI Setup
root = tk.Tk()
root.title("Climate Data Analysis")

# Labels and entries
ttk.Label(root, text="Enter today's date (e.g., '0101' for Jan 1st):").grid(column=0, row=0, padx=10, pady=5)
entry_date = ttk.Entry(root)
entry_date.grid(column=1, row=0, padx=10, pady=5)

ttk.Label(root, text="Enter today's mean temperature (°C):").grid(column=0, row=1, padx=10, pady=5)
entry_temp = ttk.Entry(root)
entry_temp.grid(column=1, row=1, padx=10, pady=5)

ttk.Label(root, text="Enter today's mean precipitation (mm):").grid(column=0, row=2, padx=10, pady=5)
entry_precip = ttk.Entry(root)
entry_precip.grid(column=1, row=2, padx=10, pady=5)

# Buttons to show results and plot PDF
ttk.Button(root, text="Compare", command=show_results).grid(column=0, row=3, columnspan=2, padx=10, pady=5)
ttk.Button(root, text="Plot PDF", command=plot_pdf).grid(column=0, row=4, columnspan=2, padx=10, pady=5)

# Result display
result_text = tk.StringVar()
ttk.Label(root, textvariable=result_text, wraplength=400).grid(column=0, row=5, columnspan=2, padx=10, pady=5)

root.mainloop()