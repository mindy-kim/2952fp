import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_unit_normal_vectors(n_samples):
    """
    Generate n_samples random 3D unit normal vectors.
    
    Args:
        n_samples (int): Number of vectors to generate.
    
    Returns:
        np.ndarray: Array of shape (n_samples, 3) with unit vectors.
    """
    # Sample points from a normal distribution
    vectors = np.random.randn(n_samples, 3)
    # Normalize to unit length
    unit_vectors = vectors
    return unit_vectors

def plot_vectors_with_epsilon_ball(vectors, epsilon, center_point):
    """
    Visualize 3D unit normal vectors and highlight points within epsilon of a center point.
    
    Args:
        vectors (np.ndarray): Array of shape (n_samples, 3) with 3D unit vectors.
        epsilon (float): Radius of the epsilon ball.
        center_point (np.ndarray): 3D point representing the center of the epsilon ball.
    """
    # Compute distances from each point to the center point
    distances = np.linalg.norm(vectors - center_point, axis=1)
    
    # Identify points inside and outside the epsilon ball
    inside_ball = distances <= epsilon
    outside_ball = ~inside_ball

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points inside the epsilon ball (red)
    ax.scatter(
        vectors[inside_ball, 0],
        vectors[inside_ball, 1],
        vectors[inside_ball, 2],
        c='red',
        label=f'Inside ε-ball (ε={epsilon})',
        alpha=0.7
    )
    
    # Plot points outside the epsilon ball (green)
    ax.scatter(
        vectors[outside_ball, 0],
        vectors[outside_ball, 1],
        vectors[outside_ball, 2],
        c='green',
        label='Outside ε-ball',
        alpha=0.7
    )
    
    # Plot the epsilon ball center point
    ax.scatter(
        center_point[0], center_point[1], center_point[2],
        c='blue', label='Epsilon Ball Center', s=100, edgecolor='black'
    )
    
    # Draw the epsilon ball
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = epsilon * np.cos(u) * np.sin(v) + center_point[0]
    y = epsilon * np.sin(u) * np.sin(v) + center_point[1]
    z = epsilon * np.cos(v) + center_point[2]
    ax.plot_wireframe(x, y, z, color='blue', alpha=0.3, linewidth=0.5)
    
    # Set plot limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Forget Task ε-ball')
    
    # Add legend outside the plot
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_samples = 1000  # Number of 3D vectors
    epsilon = 0.5     # Radius of the epsilon ball
    vectors = generate_unit_normal_vectors(n_samples)
    
    # Randomly select the center point for the epsilon ball
    center_point = vectors[np.random.randint(0, n_samples)]
    
    # Visualize the vectors and the epsilon ball
    plot_vectors_with_epsilon_ball(vectors, epsilon, center_point)