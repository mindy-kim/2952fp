import argparse
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
rc('font', size=8)
rc('font', weight='bold')
rc('text.latex', preamble=r'\usepackage{amsmath}')

def extract_train_loss(logdir):
    """
    Extracts the training loss and corresponding steps from TensorBoard logs.
    
    Args:
        logdir (str): Path to the TensorBoard log directory.
    
    Returns:
        steps (list of int): Training steps.
        losses (list of float): Corresponding training loss values.
    """
    if not os.path.exists(logdir):
        print(f"Error: The log directory '{logdir}' does not exist.")
        sys.exit(1)
    
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    if 'train_loss' not in event_acc.Tags().get('scalars', []):
        print("Error: 'train_loss' metric not found in the provided log directory.")
        sys.exit(1)
    
    events = event_acc.Scalars('train_loss')
    steps = [event.step for event in events]
    losses = [event.value for event in events]
    
    return steps, losses

def plot_loss(steps, losses, output_path=None):
    """
    Plots training loss versus steps with a minimalist black style and a title.
    
    Args:
        steps (list of int): Training steps.
        losses (list of float): Corresponding training loss values.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(6, 4))
    
    # Set minimalist black style
    plt.style.use('seaborn-whitegrid')
    plt.plot(steps, losses, color='black', linewidth=1.5)
    
    # Title
    plt.title('Baseline Transformer Training Loss', fontsize=14)
    
    # Labeling
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot training loss from TensorBoard logs.")
    parser.add_argument('logdir', type=str, help='Path to the TensorBoard log directory.')
    parser.add_argument('--output', type=str, default=None, help='Path to save the plot (e.g., loss_plot.png). If not provided, the plot will be displayed.')
    
    args = parser.parse_args()
    
    steps, losses = extract_train_loss(args.logdir)
    
    if not steps:
        print("No training loss data found.")
        sys.exit(1)
    
    plot_loss(steps, losses, args.output)

if __name__ == "__main__":
    main()