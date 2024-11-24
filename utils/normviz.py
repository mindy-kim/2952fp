import numpy as np
import matplotlib.pyplot as plt

num_samples = 10000
dimensions = 5
vectors = np.random.randn(num_samples, dimensions)

norms = np.linalg.norm(vectors, axis=1)

plt.figure(figsize=(8, 6))
plt.hist(norms, bins=50, density=True, alpha=0.7, edgecolor='black')

plt.xlabel('Norm', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('PDF of 5-Dimensional Vector Norms', fontsize=16)
plt.grid(alpha=0.3)
plt.show()