import torch
import matplotlib.pyplot as plt
from pattern import  generate_unique_filter_linear

def test_linear_pattern():
    img_sizes = [128]
    pattern_indices = [0, 1, 2]
    
    fig, axes = plt.subplots(len(img_sizes), len(pattern_indices), figsize=(12, 8))
    
    for i, img_size in enumerate(img_sizes):
        for j, pattern_idx in enumerate(pattern_indices):
            pattern = generate_unique_filter_linear(pattern_idx, img_size)
            
            if len(img_sizes) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
                
            im = ax.imshow(pattern, cmap='viridis')
            ax.set_title(f"{pattern_idx=}, {img_size=}")
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout()
    plt.savefig("linear_patterns.png")
    plt.show()

if __name__ == "__main__":
    test_linear_pattern()
    