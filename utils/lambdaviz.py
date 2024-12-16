import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
import numpy as np

# Set LaTeX-style text rendering
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
rc('font', size=8)
rc('font', weight='bold')
rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_retain_loss_lambda(data, lambda_values, n_f_values, output_path=None):
    """
    Plots retain loss vs lambda_r for different N_f values.
    
    Args:
        data (dict): Dictionary with N_f values as keys and lists of retain loss values as values.
        lambda_values (list): List of lambda_r values.
        n_f_values (list): List of N_f values.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(6, 4))
    
    # Set minimalist black style
    plt.style.use('seaborn-whitegrid')
    
    for n_f in n_f_values:
        plt.plot(lambda_values, data[n_f], marker='o', label=f'$N_f={n_f}$', linewidth=1.5)
    
    # Title
    plt.title('Retain Loss vs $\\lambda_r$ for Different $N_f$', fontsize=14)
    
    # Labeling
    plt.xlabel(r'$\lambda_r$', fontsize=12)
    plt.ylabel(r'Retain Loss', fontsize=12)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)
    
    # Legend
    plt.legend(title=r'$N_f$', fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    # Example data
    lambda_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_f_values = [1, 2, 4, 8]
    
    # Simulated retain loss values for each N_f
    data = {
        1: [3.420, 0.255, 0.212, 0.178, 0.124, 0.074],
        2: [3.300, 0.302, 0.314, 0.210, 0.184, 0.071],
        4: [3.380, 0.462, 0.324, 0.241, 0.204, 0.073],
        8: [3.360, 0.628, 0.464, 0.338, 0.243, 0.073],
    }
    
    output_path = 'retain_loss_lambda_plot.png'  # Set to None to display the plot instead
    plot_retain_loss_lambda(data, lambda_values, n_f_values, output_path)

if __name__ == "__main__":
    main()