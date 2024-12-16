import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams

rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
rc('font', size=8)
rc('font', weight='bold')
rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_loss(data_points, output_path=None):
    """
    Plots retain loss versus number of forget tasks with a minimalist black style.
    
    Args:
        data_points (list of tuples): List of (num_forget_tasks, retain_loss) pairs.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    num_tokens, retain_loss, forget_loss = zip(*data_points)

    plt.figure(figsize=(6, 4))
    
    # Set minimalist black style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(num_tokens, retain_loss, marker='o', color='blue', label=f'Retain Loss', linewidth=1.5)
    plt.plot(num_tokens, forget_loss, marker='o', color='red', label=f'Forget Loss', linewidth=1.5)

    # Title
    plt.title('Adversarial Loss vs No. of Forget Tasks Tradeoff', fontsize=14)

    plt.legend(title=r'Adversarial Loss', fontsize=10, loc='upper left')

    # Labeling
    plt.xlabel(r'Number of Forget Tasks', fontsize=12)
    plt.ylabel(r'Loss', fontsize=12)
    
    plt.ylim(0, max(retain_loss + forget_loss) * 1.1)  # Add some padding at the top for clarity

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
    # Example manually provided data points
    data_points = [
        (1, 3.326, 3.370),
        (2, 1.932, 1.784),
        (4, 2.057, 1.498),
        (8, 1.396, 3.071),
    ]
    
    output_path = 'hijack_nf.png'  # Set to None to display the plot instead
    plot_loss(data_points, output_path)

if __name__ == "__main__":
    main()