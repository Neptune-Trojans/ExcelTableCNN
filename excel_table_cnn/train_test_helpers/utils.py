import numpy as np
import torch


def compute_feature_map_aspect_ratios(feature_maps):
    """
    Computes the aspect ratio (width / height) for each table feature map.

    Args:
        feature_maps (List[Tensor]): Each tensor is of shape (H, W, C)

    Returns:
        List[float]: Aspect ratios (W / H) for each feature map
    """
    aspect_ratios = []

    for fm in feature_maps:
        height, width = fm.shape[0], fm.shape[1]
        aspect_ratio = width / max(height, 1e-6)  # avoid division by zero
        aspect_ratios.append(aspect_ratio)

    aspect_ratios = np.array(aspect_ratios)  # paste your values here

    # Core stats
    print("Min:", np.min(aspect_ratios))
    print("Max:", np.max(aspect_ratios))
    print("Mean:", np.mean(aspect_ratios))
    print("Median:", np.median(aspect_ratios))
    print("Quantiles:", np.quantile(aspect_ratios, [0.25, 0.5, 0.75, 0.9, 0.95]))

    # Distribution buckets
    tall = np.sum(aspect_ratios < 0.5)
    square = np.sum((aspect_ratios >= 0.5) & (aspect_ratios <= 1.5))
    wide = np.sum(aspect_ratios > 1.5)

    print(f"Tall (<0.5): {tall} examples")
    print(f"Square-ish (0.5–1.5): {square} examples")
    print(f"Wide (>1.5): {wide} examples")

    return aspect_ratios

def unique_feature_vectors_with_counts(x):
    """
    Given a tensor of shape (N, M, 17), returns unique 17D feature vectors and their occurrence counts.

    Args:
        x (torch.Tensor): Input tensor of shape (N, M, 17)

    Returns:
        unique_vectors (Tensor): Unique feature vectors (K, 17)
        counts (Tensor): Counts of each unique vector (K,)
    """
    # Flatten to (N*M, 17)
    x_flat = x.view(-1, x.shape[-1])

    # Use torch.unique with return_counts
    unique_vectors, counts = torch.unique(x_flat, dim=0, return_counts=True)
    for vector, count in zip(unique_vectors, counts):
        print(f'{vector} count:{count}')

    return unique_vectors, counts