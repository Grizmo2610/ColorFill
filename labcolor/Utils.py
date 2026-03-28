import torch

import numpy as np

import matplotlib.pyplot as plt

import re
import os
import random
import math
import json

def seed_everything(seed):
    # Set fixed seed for reproducibility across libraries
    random.seed(seed)                          # Python random module seed
    np.random.seed(seed)                       # NumPy seed
    torch.manual_seed(seed)                    # PyTorch CPU seed
    torch.cuda.manual_seed(seed)               # PyTorch CUDA seed for single GPU
    torch.cuda.manual_seed_all(seed)           # PyTorch CUDA seed for all GPUs if using multi-GPU
    # Ensure deterministic behavior for CUDA convolution operations
    torch.backends.cudnn.deterministic = True
    # Disable benchmark mode to prevent nondeterministic algorithm selection
    torch.backends.cudnn.benchmark = False

def init_history(keys: list[str] | None = None) -> dict[str, dict[str, list]]:
    if keys is None:
        keys = ["loss"]  # default keys

    return {phase: {key: [] for key in keys} for phase in ["train", "val"]}

def plot_history(
    history: dict[str, dict[str, list]],
    paths: dict = {},
    save: bool = True,
    root: str = "sample"
):
    os.makedirs(root, exist_ok=True)
    
    # Extract training and validation history
    train_history = history['train']
    val_history = history['val']
    
    # Generate epoch indices
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Define full paths for saving plot image and history file
    history_path = os.path.join(root, paths.get("history", "history.json"))
    plot_image_path = os.path.join(root, paths.get("plot_image", "history_plot.png"))

    n_metrics = len(train_history)
    cols = 2
    rows = math.ceil(n_metrics / cols)    
    
    # Create a figure with subplots for each metric
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, k in enumerate(train_history):
        plt.subplot(rows, cols, i + 1)
        plt.plot(epochs, train_history[k], 'bo-', label=f'Train {k.capitalize()}')
        plt.plot(epochs, val_history[k], 'ro-', label=f'Val {k.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(k.capitalize())
        plt.title(f'{k.capitalize()} over Epochs')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    
    # Save plot
    if save:
        plt.savefig(plot_image_path)
        # Save history as JSON
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
    
    plt.show()