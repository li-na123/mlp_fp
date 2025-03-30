from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_sample(file_path: Path):
    data = np.load(file_path)
    embeddings = data['embeddings']
    labels = data['labels']
    return embeddings, labels

sample_file = Path("embeddings/train/sample_000006.npz")
embeddings, labels = load_sample(sample_file)

print(f"Embeddings shape: {embeddings.shape}")  # e.g., (seq_length, hidden_size)
print(f"Aligned labels: {labels}")
print(f"Mean embedding vector: {np.mean(embeddings, axis=0)}")
print(f"Std of embeddings: {np.std(embeddings, axis=0)}")

# We'll reduce the embedding of one sample (for example, its tokens) to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Optionally, annotate some of the points with their labels
for i, label in enumerate(labels):
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
    
plt.title("PCA of Sample Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
