import torch


def collate(samples):
    batch = {
        "image": torch.stack([s["image"] for s in samples], dim=0),
        "mask": torch.stack([s["mask"] for s in samples], dim=0),
        "name": [s["name"] for s in samples],
    }
    return batch