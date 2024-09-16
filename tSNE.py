import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from load_data import load_vit_data

_, _, data, labels, _ = load_vit_data(state=5)

classes =list(np.unique(labels))
print(classes)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10)

plt.colorbar(scatter, ticks=classes, label='Class Labels') 
plt.title("t-SNE Results Colored by Class Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
