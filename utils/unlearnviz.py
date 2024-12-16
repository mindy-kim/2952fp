import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
rc('font', size=8)
rc('font', weight='bold')
rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_retain_loss(data_points, output_path=None):
    """
    Plots retain loss versus number of forget tasks with a minimalist black style.
    
    Args:
        data_points (list of tuples): List of (num_forget_tasks, retain_loss) pairs.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    num_forget_tasks, retain_loss = zip(*data_points)

    plt.figure(figsize=(6, 4))
    
    # Set minimalist black style
    plt.style.use('seaborn-whitegrid')
    plt.plot(num_forget_tasks, retain_loss, marker='o', color='black', linewidth=1.5)
    
    # Title
    plt.title('Retain Loss vs No. of Forget Tasks Tradeoff', fontsize=14)
    
    # Labeling
    plt.xlabel(r'Number of Forget Tasks', fontsize=12)
    plt.ylabel(r'Retain Loss', fontsize=12)
    
    plt.ylim(0, max(retain_loss) * 1.1)  # Add some padding at the top for clarity

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
        (1, 0.212),
        (2, 0.302),
        (4, 0.462),
        (8, 0.628),
    ]
    
    output_path = 'retain_loss_plot.png'  # Set to None to display the plot instead
    plot_retain_loss(data_points, output_path)

if __name__ == "__main__":
    main()




    data_points = [
        (1, 0.212),
        (2, 0.302),
        (4, 0.462),
        (8, 0.628),
    ]