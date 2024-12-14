import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

for dimensions in [10, 100, 1000]:
    num_samples = 10000
    vectors = np.random.randn(num_samples, dimensions)

    norms = np.linalg.norm(vectors, axis=1)

    plt.hist(norms, bins=50, density=True, alpha=0.6, label=f'{dimensions} Dimensions', edgecolor='black')

plt.xlabel('Norm', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('PDF of Vector Norms for Different Dimensions', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.show()