import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class ImageNet(Dataset):
    def __init__(self, root: str, transform=None, p: float = 0.5, max_hint: float = 0.1):
        self.root = root
        self.transform = transform
        self.p = p
        self.max_hint = max_hint
        self.paths = []
        all_dirs = list(os.walk(root))
        loop = tqdm(all_dirs, desc="Adding data")

        for dirpath, dirnames, filenames in loop:
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.paths.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = cv2.imread(path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy() * 255  # CHW → HWC, float → 0-255
                image = image.astype(np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0:1] / 50.0 - 1.0
        ab = (lab[:, :, 1:] - 128.0) / 110.0

        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        L_tensor = torch.from_numpy(L).permute(2, 0, 1)
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)

        hint, mask = self._make_hint(ab_tensor)

        return {
            "image": image_tensor,
            "L": L_tensor,
            "ab": ab_tensor,
            "hint": hint,
            "mask": mask
        }

    def _make_hint(self, ab: torch.Tensor):
        if torch.rand(1).item() > self.p:
            return torch.zeros_like(ab), torch.zeros(1, ab.shape[1], ab.shape[2])

        hint = torch.zeros_like(ab)
        mask = torch.zeros(1, ab.shape[1], ab.shape[2])
        num_points = int(self.max_hint * ab.shape[1] * ab.shape[2])
        ys = torch.randint(0, ab.shape[1], (num_points,))
        xs = torch.randint(0, ab.shape[2], (num_points,))
        hint[:, ys, xs] = ab[:, ys, xs]
        mask[:, ys, xs] = 1.0
        return hint, mask