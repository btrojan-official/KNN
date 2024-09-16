import numpy as np
import umap
import matplotlib.pyplot as plt

from load_data import load_vit_data

_, _, data, labels, _ = load_vit_data(state=5)

classes =list(np.unique(labels))
print(classes)

umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_results = umap_model.fit_transform(data)

# Step 4: Plot the UMAP results, coloring by labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='viridis', s=10)

# Step 5: Add colorbar for classes and display plot
plt.colorbar(scatter, ticks=classes, label='Class Labels')  # Adjust ticks based on your number of classes
plt.title("UMAP Results Colored by Class Labels")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.show()
