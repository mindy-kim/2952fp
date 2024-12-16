import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
import numpy as np
from itertools import cycle

def extract_data(logdir, filepath, metrics):
    """
    Extracts metrics data.
    
    Args:
        logdir (str): Path to the log directory.
        metrics (list): List of metric tags to extract.
    
    Returns:
        dict: A dictionary where keys are metric names, and values are lists of (step, value) tuples.
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
    except KeyError:
        print(f"Metric file not found in {logdir}")

    for metric in metrics:
        data[metric] = [(step, value) for step, value in data[metric].items()]

    print(data['loss/retain_loss'][-1])
    print(data['loss/forget_loss'][-1])

    return data

def plot_metrics_nf(data_by_task, output_file=None):
    """
    Plots metrics for multiple tasks with professional styling.
    
    Args:
        data_by_task (dict): A dictionary where keys are task identifiers, and values are dictionaries of metrics data.
        output_file (str): Optional. If provided, saves the plot to this file.
    """
    plt.figure(figsize=(12, 8))
    
    # Define professional color palette and line styles
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    line_styles = cycle(["--", "-"])
    
    for task_id, task_data in data_by_task.items():
        color = next(colors)
        for metric, values in task_data.items():
            steps, metrics_values = zip(*values)
            linestyle = next(line_styles)
            
            label = f"{task_id} - {metric.replace('loss/', '').replace('_', ' ').capitalize()}"
            plt.plot(
                steps,
                metrics_values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
    
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Hijacking Retain/Forget Loss", fontsize=16)
    plt.legend(fontsize=12, loc="center right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def plot_metrics_token(data_by_task, output_file=None):
    """
    Plots metrics for multiple tasks with professional styling.
    
    Args:
        data_by_task (dict): A dictionary where keys are task identifiers, and values are dictionaries of metrics data.
        output_file (str): Optional. If provided, saves the plot to this file.
    """
    plt.figure(figsize=(12, 8))
    
    # Define professional color palette and line styles
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    line_styles = cycle(["--", "-"])
    
    for task_id, task_data in data_by_task.items():
        color = next(colors)
        steps, metrics_values = zip(*task_data['loss/retain_loss'])
        linestyle = next(line_styles)
        
        label = f"# Tokens: {task_id}"
        plt.plot(
            steps,
            metrics_values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    
    plt.xlabel("Number of Tokens", fontsize=14)
    plt.ylabel("Retain Loss", fontsize=14)
    plt.title("Retain Loss - Tokens Tradeoff", fontsize=16)
    plt.legend(fontsize=12, loc="center right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create professional plots from TensorBoard logs for paper-ready figures.")
    parser.add_argument("logdirs", nargs="+", type=str, help="List of TensorBoard log directories for different tasks.")
    parser.add_argument("--task_names", nargs="+", type=str, help="Names of tasks corresponding to the log directories.")
    parser.add_argument("--output", type=str, default=None, help="Output file for the plot (e.g., plot.png).")
    parser.add_argument("--tokens", type=bool, default=False, help="If comparing number of tokens or number of tasks")
    args = parser.parse_args()
    
    if not (len(args.logdirs) == len(args.task_names) or (len(args.logdirs) == 1 and args.tokens)):
        raise ValueError("The number of log directories must match the number of task names or use only one log directory.")
    
    metrics = ["loss/forget_loss", "loss/retain_loss"]
    files = ['metrics1.pkl', 'metrics5.pkl', 'metrics10.pkl', 'metrics20.pkl']
    data_by_task = {}
    
    if not args.tokens:
        for logdir, task_name in zip(args.logdirs, args.task_names):
            print(f"Extracting data for task: {task_name}")
            filepath = os.path.join(logdir, 'metrics1.pkl')
            task_data = extract_data(logdir, filepath, metrics)
            data_by_task[task_name] = task_data

        print("Plotting metrics...")
        plot_metrics_nf(data_by_task, output_file=args.output)
    else:
        for i, (file, task_name) in enumerate(zip(files, args.task_names)):
            print(f"Extracting data for task: {task_name}")
            filepath = os.path.join(args.logdirs[i], file)
            task_data = extract_data(args.logdirs[i], filepath, metrics)
            data_by_task[task_name] = task_data

        print("Plotting metrics...")
        plot_metrics_token(data_by_task, output_file=args.output)

    # python utils/hijackviz.py logs/hijack1_full_lsamlp/ logs/hijack2_full_lsamlp/ logs/hijack4_full_lsamlp/ logs/hijack8_full_lsamlp/ --task_names 1 2 4 8 --output figures/hijack_nf
    # logs/hijack8_full_lsamlp/ 
    # python utils/hijackviz.py logs/hijack1_full_lsamlp/ logs/hijack1_5_full_lsamlp/ logs/hijack1_10_full_lsamlp/ logs/hijack1_20_full_lsamlp/ --task_names 1 5 10 20 --output figures/hijack_token --tokens True