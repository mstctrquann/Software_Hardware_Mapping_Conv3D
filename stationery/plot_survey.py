import os
import sys
import io
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def autolabel(rects, ax, format_str='%.2e'):
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate(format_str % height if format_str else str(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

def plot_metric(labels, values, title, ylabel, filepath, format_str='%.2e'):
    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    colors = ['gray', 'royalblue', 'forestgreen', 'firebrick']
    
    # Handle zero values for log scale
    has_zero = any(v == 0 for v in values)
    
    if has_zero:
        bars = plt.bar(x, values, color=colors, edgecolor='black')
        plt.yscale('symlog', linthresh=1)
        note_text = "Scale: Symmetrical Log\nFormula: y = sign(x) * log10(1 + |x|)"
    else:
        # Set bottom to avoid log(0) issues when drawing bars
        min_positive = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1
        bottom_val = 1 if min_positive >= 1 else min_positive / 10
        bars = plt.bar(x, values, color=colors, edgecolor='black', bottom=bottom_val)
        plt.yscale('log')
        note_text = "Scale: Logarithmic (Base 10)\nFormula: y = log10(x)"

    # Tăng giới hạn trục Y để tạo khoảng trống phía trên cho annotation box và text
    max_val = max(values) if values else 1
    if max_val > 0:
        plt.ylim(top=max_val * 500)
    else:
        plt.ylim(top=100)

    plt.ylabel(ylabel + " (Log)", fontsize=11)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, labels, fontsize=11)
    
    # Add formula note in the corner
    plt.annotate(note_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
                 verticalalignment='top')
        
    autolabel(bars, plt.gca(), format_str)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def main():
    base_dir = "Results"
    csv_path = os.path.join(base_dir, "survey_results.csv")
    plot_dir = os.path.join(base_dir, "survey_plots")
    ensure_dir(plot_dir)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    labels = []
    metrics_data = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # Initialize metrics lists
        for h in headers[1:]:
            metrics_data[h] = []
            
        for row in reader:
            labels.append(row[0])
            for i, h in enumerate(headers[1:]):
                metrics_data[h].append(float(row[i+1]))

    print(f"Generating plots in {plot_dir}...")

    # Plot each metric
    for metric_name, values in metrics_data.items():
        # Determine format based on metric type
        if "Utilization" in metric_name:
            format_str = "%.2f%%"
            ylabel = "Percentage (%)"
        elif "Calls" in metric_name or "MACs" in metric_name:
            format_str = "%.2e"
            ylabel = "Count"
        elif "Cycles" in metric_name:
            format_str = "%.2e"
            ylabel = "Cycles"
        else:
            format_str = "%.2e"
            ylabel = "Value"
            
        safe_name = metric_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").lower()
        filepath = os.path.join(plot_dir, f"{safe_name}.png")
        plot_metric(labels, values, metric_name, ylabel, filepath, format_str)
        print(f"Saved: {filepath}")

    print("All survey plots generated successfully.")

if __name__ == "__main__":
    main()
