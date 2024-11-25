import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from itertools import cycle

def extract_data(logdir, metrics):
    """
    Extracts scalar data from TensorBoard logs.
    
    Args:
        logdir (str): Path to the TensorBoard log directory.
        metrics (list): List of metric tags to extract.
    
    Returns:
        dict: A dictionary where keys are metric names, and values are lists of (step, value) tuples.
    """
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    data = {}
    for metric in metrics:
        try:
            scalar_data = event_acc.Scalars(metric)
            data[metric] = [(entry.step, entry.value) for entry in scalar_data]
        except KeyError:
            print(f"Metric '{metric}' not found in {logdir}")
    
    return data

def plot_metrics(data_by_task, output_file=None):
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
    
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("LSA-MLP Retain/Forget Loss", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
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
    args = parser.parse_args()
    
    if len(args.logdirs) != len(args.task_names):
        raise ValueError("The number of log directories must match the number of task names.")
    
    metrics = ["loss/forget_loss", "loss/retain_loss"]
    data_by_task = {}
    
    for logdir, task_name in zip(args.logdirs, args.task_names):
        print(f"Extracting data for task: {task_name}")
        task_data = extract_data(logdir, metrics)
        data_by_task[task_name] = task_data
    
    print("Plotting metrics...")
    plot_metrics(data_by_task, output_file=args.output)