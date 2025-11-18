import json, torch
from pathlib import Path
from PIL import Image
import numpy as np

class RoadGraphDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.items = list(files)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        jpath = self.items[idx]
        with open(jpath) as f:
            J = json.load(f)

        # image
        ipath = Path(J["image_path"])
        img = Image.open(ipath).convert("RGB")
        W = H = J["crop"]["out_size"]
        image = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 # C, H, W

        # nodes 
        nodes_xy = []
        nodes_global_ids = [] 
        for n in J["nodes"]:
            gid = int(n["idx"])
            x = float(n["x"])
            y = float(n["y"])
            # clamp tiny negatives
            x = min(max(x, 0.0), W - 1.0)
            y = min(max(y, 0.0), H - 1.0)
            nodes_xy.append((x, y))
            nodes_global_ids.append(gid)
        nodes = torch.tensor(nodes_xy, dtype=torch.float32) if nodes_xy else torch.zeros((0, 2), dtype=torch.float32)
        nodes_global_ids = torch.tensor(nodes_global_ids, dtype=torch.long)

        # edges
        e_idx = []
        for e in J["edges"]:
            e_idx.append((int(e["src_idx"]), int(e["dst_idx"])))
        edges = torch.tensor(e_idx, dtype=torch.float32) if e_idx else torch.zeros((0, 2), dtype=torch.float32)

        sample = {
            "image": image,
            "nodes": nodes,
            "edges": edges,
            "node_ids": nodes_global_ids,
            "meta": {
                "tile_id": ipath.stem,
                "H": H,
                "W": W,
                "coord_convention": J.get("coord_convention", "pixel_y_down"),
                "region": J.get("region", ""),
                "json_path": str(jpath),
                "image_path": str(ipath)
            }
        }
        return sample