import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")



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