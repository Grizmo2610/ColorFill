import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class ImageNet(Dataset):
    def __init__(self, root: str, transform=None, p: float = 0.5, max_hint: float = 0.1):
        self.root = root
        self.transform = transform
        self.p = p
        self.max_hint = max_hint
        self.paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                self.paths.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = np.array(Image.open(path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        L = lab[:, :, 0:1] / 50.0 - 1.0          # normalize L ~ [-1,1]
        ab = (lab[:, :, 1:] - 128.0) / 110.0    # normalize ab ~ [-1,1]

        # to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        L = torch.from_numpy(L).permute(2, 0, 1)
        ab = torch.from_numpy(ab).permute(2, 0, 1)

        hint, mask = self._make_hint(ab)

        return {
            "image": image,
            "L": L,
            "ab": ab,
            "hint": hint,
            "mask": mask
        }

    def _make_hint(self, ab: torch.Tensor):
        if torch.rand(1).item() > self.p:
            return torch.zeros_like(ab), torch.zeros(1, ab.shape[1], ab.shape[2])

        ratio = torch.rand(1).item() * self.max_hint
        mask = (torch.rand(1, ab.shape[1], ab.shape[2]) < ratio).float()
        hint = ab * mask
        return hint, mask