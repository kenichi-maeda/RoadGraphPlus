import torch


def collate(samples):
    batch = {
        "image": torch.stack([s["image"] for s in samples], 0),
        "nodes": [s["nodes"] for s in samples],
        "edges": [s["edges"] for s in samples],
        "meta": [s["meta"] for s in samples]
    }
    return batch