import os
import shutil
import time
import random
import re
import math
import errno

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

from labcolor import *

config = {
    "seed": 42,
    "ROOT": "/kaggle/input/imdb-clean",
    
    "MODEL_FOLDER": "models",
    "SAMPLE_FOLDER": "sample",
    "plot_image_path": "plot_image.png",
    "history_path": "history.json",
    "model_path": "models/best_delta_E_model.pth",
    "BATCH_SIZE": 16,
    "EPOCHS": 10,
    "IMG_SIZE": 224,
    "LR": 3e-4,
    "ratio": {"train": 0.8, "val": 0.2}
}

criterion = {
    "mse_loss": F.mse_loss,
}


seed_everything(config["seed"])  # Set random seed for reproducibility

# Select device: use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Number of CPU cores for data loading parallelism
num_workers = os.cpu_count()

print(f"Device: {device}")
print(f"Number of CPU cores: {num_workers}")

# Flag to indicate whether to clear old data before running
clear = True

if clear:
    for root in (config["MODEL_FOLDER"], config["SAMPLE_FOLDER"]):
        try:
            shutil.rmtree(root)
        except Exception:
            pass

        try:
            os.makedirs(root, exist_ok=True)
        except Exception:
            pass


def build_transforms(IMG_SIZE: int = 224):
    train_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        # Biến đổi hình học
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.5),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),

        # Biến đổi màu/ảnh grayscale (L channel)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        
        # Color jitter / ab perturb: dùng HueSaturationValue như proxy cho ab
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),

        # To tensor
        ToTensorV2()
    ])

    # Validation transform: chỉ resize/crop, không augment
    val_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        ToTensorV2()
    ])
    

    return train_transform, val_transform


train_transform, val_transform = build_transforms(config["IMG_SIZE"])

dataset = ImageNet(config["ROOT"])

train_size = max(config["ratio"]["train"])
val_size = min(config["ratio"]["val"])

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=num_workers, pin_memory=True)

best_metrics = {
    "loss": float("inf"),
    "delta": float("inf")
}
default_keys = ['loss', 'delta']
history = init_history(default_keys)

model = NeuralColor()

for p in model.net.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    [{"params": model.net.parameters(), "lr": config["LR"]}],
    weight_decay=0.01)

loader = {"train": train_loader, "val": val_loader}
history = fit(
    model=model, 
    optimizer=optimizer, 
    device=device, 
    epochs=config["EPOCHS"], 
    criterion=criterion, 
    loader=loader,
    history=history, 
    best_metrics=best_metrics,
)

for p in model.net.parameters():
    p.requires_grad = True
    
history = fit(
    model=model, 
    optimizer=optimizer, 
    device=device, 
    epochs=config["EPOCHS"], 
    criterion=criterion, 
    loader=loader,
    history=history, 
    best_metrics=best_metrics,
)

plot_history(history, {"plot_image": config["plot_image_path"], "history": config["history_path"]}, root=config["SAMPLE_FOLDER"])