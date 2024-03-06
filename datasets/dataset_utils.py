import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.utils import make_grid

def idx_to_float(idx: np.ndarray, num_bins: int):
    flt_zero_one = (idx + 0.5) / num_bins
    return (2.0 * flt_zero_one) - 1.0

def float_to_idx(flt: np.ndarray, num_bins: int):
    flt_zero_one = (flt / 2.0) + 0.5
    return torch.clamp(torch.floor(flt_zero_one * num_bins), min=0, max=num_bins - 1).long()

def quantize(flt, num_bins: int):
    return idx_to_float(float_to_idx(flt, num_bins), num_bins)

def get_image_grid_from_tensor(image_tensor, nrows=1):
    return make_grid(image_tensor, nrow=nrows, normalize=True)

def plot_tensor_images(images, n=8):
    fig, axes = plt.subplots(1, n, figsize=(10, 10))
    for i in range(n):
        image = images[i].squeeze().cpu().detach().numpy()
        image = (image + 1) / 2  # Rescale from (-1, 1) to (0, 1)
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
    plt.show()

