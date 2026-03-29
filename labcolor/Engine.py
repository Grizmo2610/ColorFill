import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import time
import os
import re
from typing import Optional

def denorm_lab(L, ab):
    L = L * 100.0
    ab = ab * 128.0
    return L, ab

def compute_deltaE(pred_ab, gt_ab, L):
    # scale lại
    L, pred_ab = denorm_lab(L, pred_ab)
    _, gt_ab   = denorm_lab(L, gt_ab)

    # ghép thành Lab
    pred_lab = torch.cat([L, pred_ab], dim=1)
    gt_lab   = torch.cat([L, gt_ab], dim=1)

    # deltaE
    deltaE = torch.sqrt(((pred_lab - gt_lab) ** 2).sum(dim=1))  # (B,H,W)
    
    return deltaE.mean()

def get_latest_epoch(
    root: str = 'models', 
    prefix="model_epoch_", 
    suffix=".pth"
) -> int:
    max_epoch = 0  # Initialize max epoch to 0

    # Iterate through all files in the specified folder
    for filename in os.listdir(root):
        # Check if the filename matches the pattern "model_epoch_{number}.pth"
        match = re.match(fr"{prefix}(\d+){suffix}", filename)
        if match:
            # Extract the epoch number from the filename
            epoch_num = int(match.group(1))
            # Update max_epoch if this epoch number is greater
            if epoch_num > max_epoch:
                max_epoch = epoch_num

    return max_epoch  # Return the highest epoch number found

def save_best_models(
    model: nn.Module,
    val_result: dict[str, float],
    epoch: int,
    save_paths: dict[str, str],
    root: str = "models",
    best_metrics: dict[str, float] | None = None,
) -> bool:

    os.makedirs(root, exist_ok=True)

    if best_metrics is None:
        best_metrics = {}

    # Nếu DataParallel, lấy model gốc
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    updated = False

    for metric, value in val_result.items():

        if metric not in save_paths:
            continue

        save_path = os.path.join(root, save_paths[metric])

        best_val = best_metrics.get(metric, float("inf"))

        if value < best_val:
            best_metrics[metric] = value
            torch.save(model_to_save.state_dict(), save_path)
            print(f"✅✅Best {metric} model saved at epoch {epoch+1:02d}")
            updated = True

    return updated


def save_epoch_model(
    model: nn.Module,
    epoch: int,
    root: str="models"
):
    os.makedirs(root, exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    path = os.path.join(root, f"model_epoch_{epoch + 1:02d}.pth")
    torch.save(model_to_save.state_dict(), path)
    print(f"Model for epoch {epoch + 1:02d} saved.")

def forward_pass(loop, model, device):
    images = loop["image"].to(device, non_blocking=True)
    L = loop["L"].to(device, non_blocking=True)
    ab = loop["ab"].to(device, non_blocking=True)
    hint = loop["hint"].to(device, non_blocking=True)
    mask = loop["mask"].to(device, non_blocking=True)

    x = torch.cat([L, hint, mask], dim=1)
    outputs = model(x)

    return images, L, ab, outputs


def compute_metrics(outputs, ab, L, loss_fn):
    loss = loss_fn(outputs, ab)
    deltaE = compute_deltaE(outputs, ab, L)
    return loss, deltaE


def update_stats(total_loss: float, total_deltaE: float, total_samples: int, 
                 loss, deltaE, batch_size):
    total_loss += loss.item() * batch_size
    total_deltaE += deltaE.item() * batch_size
    total_samples += batch_size
    return total_loss, total_deltaE, total_samples


def finalize_stats(total_loss: float, total_deltaE: float, total_samples: int):
    return {
        "loss": total_loss / total_samples,
        "deltaE": total_deltaE / total_samples,
    }
    
def train(model: nn.Module, 
          train_loader: DataLoader, 
          optimizer: torch.optim.Adam, 
          device: torch.device, 
          criterion: dict):
    model.train()

    total_loss = 0.0
    total_deltaE = 0.0
    total_samples = 0

    loss_fn = criterion["mse_loss"]

    loops = tqdm(train_loader, desc="Training", leave=False)

    
    for i, loop in enumerate(loops):
        images, L, ab, outputs = forward_pass(loop, model, device)

        optimizer.zero_grad()

        loss = loss_fn(outputs, ab)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, deltaE = compute_metrics(outputs, ab, L, loss_fn)

        batch_size = images.size(0)

        total_loss, total_deltaE, total_samples = update_stats(
            total_loss, total_deltaE, total_samples,
            loss, deltaE, batch_size
        )

        loops.set_postfix(loss=loss.item(), deltaE=deltaE.item())

    return finalize_stats(total_loss, total_deltaE, total_samples)

@torch.no_grad()
def evaluate(model: nn.Module, 
             data_loader: DataLoader, 
             device: torch.device, 
             criterion: dict):
    model.eval()

    total_loss = 0.0
    total_deltaE = 0.0
    total_samples = 0

    loss_fn = criterion["mse_loss"]

    loops = tqdm(data_loader, desc="Evaluating", leave=False)

    for i, loop in enumerate(loops):
        images, L, ab, outputs = forward_pass(loop, model, device)

        loss, deltaE = compute_metrics(outputs, ab, L, loss_fn)

        batch_size = images.size(0)

        total_loss, total_deltaE, total_samples = update_stats(
            total_loss, total_deltaE, total_samples,
            loss, deltaE, batch_size
        )

        loops.set_postfix(loss=loss.item(), deltaE=deltaE.item())

    return finalize_stats(total_loss, total_deltaE, total_samples)

def fit(model: nn.Module, optimizer: torch.optim.Adam,
        device: torch.device, epochs: int,
        loader: dict["str", DataLoader], criterion: dict = {},
        gamma: float = 0.5, patience: int = 6,
        save_paths: dict[str, str] = {
            "loss":"best_loss_model.pth",
            "deltaE":"best_delta_model.pth"
        },
        history: dict[str, dict[str, list]] = {},
        roots: dict[str, str] = {"model": "models", "sample": "sample"},
        best_metrics: dict = {"loss": float("inf")},
        ):

    start_epoch = get_latest_epoch(roots["model"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=gamma, patience=patience//2)
    early_stop_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        start_time: float = time.time()
        print('=' * 25 + f'Epoch {epoch + 1:02d}/{start_epoch + epochs:02d}' + '=' * 25)
        train_loader = loader["train"]
        train_result: dict[str, dict[str, float]] = train(model, train_loader, optimizer, device, criterion)
        if "val" in loader.keys():
            val_loader = loader["val"]
            val_result: Optional[dict[str, dict[str, float]]] = evaluate(model, val_loader, device, criterion)
        else:
            val_result = None
        keys = list(history['train'].keys())
        
        for k in keys:
            # ---- train history ----
            history['train'][k].append(train_result[k])

            # ---- Validation history ----
            if "val" in loader.keys() and val_result is not None:
                history['val'][k].append(val_result[k])
            else:
                history['val'][k].append(-1)
        if "val" in loader.keys() and val_result is not None:
            if save_best_models(model, val_result, epoch, save_paths, roots["model"], best_metrics):
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f'Stopping counter {early_stop_counter}/{patience}')
        save_epoch_model(model, epoch, roots["model"])
    
        minutes, seconds = divmod(time.time() - start_time, 60)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch time: {int(minutes):02d}:{int(seconds):02d}"
            f"LR: {current_lr:.6f}")

        # ===== LOSS =====
        if "val" in loader.keys() and val_result is not None:
            print(
                f"Train Loss: {train_result['loss']:.4f} | "
                f"Val Loss: {val_result['loss']:.4f} | "
            )
            print(
                # Overal Delta E
                f"Train ΔE: {train_result['deltaE']:.4f} | "
                f"Val ΔE: {val_result['deltaE']:.4f} | "
            )
        else:
            print(
                f"Train Loss: {train_result['loss']:.4f} | "
            )
            print(
                # Overal Delta E
                f"Train ΔE: {train_result['rmse']:.4f} | "
            )

            
        if "val" in loader.keys() and val_result is not None:
            if early_stop_counter >= patience:
                print("❌❌Early stopping triggered")
                scheduler.step(val_result['rmse'])
                return history

    return history