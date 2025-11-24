import json, torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, files=None, augment=False):
        self.items = sorted(files)
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize((512, 512), interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize((512, 512), interpolation=Image.NEAREST)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if augment:
            self.aug_color = transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3
            )
            self.blur = transforms.GaussianBlur(kernel_size=5)
        else:
            self.aug_color = None
            self.blur = None
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        sat_path = self.items[idx]
        base = sat_path.stem.replace("_sat", "")
        mask_path = sat_path.parent / f"{base}_mask.png"

        # image
        image = Image.open(sat_path).convert("RGB")
        image = self.resize(image)
        image = self.to_tensor(image)

        if self.augment:
            image = self.aug_color(image)
            image = torch.clamp(image, 0.0, 1.0)
            image = self.blur(image)

        image = self.normalize(image)

        # mask
        mask = Image.open(mask_path).convert("L")
        mask = self.resize_mask(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0
        
        sample = {
            "image": image,
            "mask": mask,
            "name": base
        }
        return sample