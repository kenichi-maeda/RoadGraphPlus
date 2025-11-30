import json, torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import math

class RoadGraphDataset(torch.utils.data.Dataset):
    def __init__(self, files, augment=False, add_vitrutal=False):
        self.items = list(files)
        self.augment = augment
        self.add_virtual = add_vitrutal

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

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
        if self.augment:
            aug = transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3
            )
            image = aug(image)
            image = torch.clamp(image, 0.0, 1.0)
            image = image + 0.02 * torch.rand_like(image)
            image = torch.clamp(image, 0.0, 1.0)
            image = transforms.GaussianBlur(kernel_size=5)(image)
            image = torch.clamp(image, 0.0, 1.0)

        image = self.normalize(image)

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

        # edges
        e_idx = []
        for e in J["edges"]:
            e_idx.append((int(e["src_idx"]), int(e["dst_idx"])))
        

        if self.add_virtual:
            # Add virtual points
            virtual_threshold = 120.0
            new_nodes = list(nodes_xy)
            new_edges = []
            new_global_ids = list(nodes_global_ids)

            max_gid = max(new_global_ids) if len(new_global_ids) > 0 else -1

            for src, dst in e_idx:
                x1, y1 = nodes_xy[src]
                x2, y2 = nodes_xy[dst]

                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx**2 + dy**2)

                if dist > virtual_threshold:
                    midpoint_x = (x1 + x2) / 2
                    midpoint_y = (y1 + y2) / 2
                    vid = len(new_nodes)
                    new_nodes.append((midpoint_x, midpoint_y))

                    max_gid += 1
                    new_global_ids.append(max_gid)

                    new_edges.append((src, vid))
                    new_edges.append((vid, dst))
                else:
                    new_edges.append((src, dst))

            nodes = torch.tensor(new_nodes, dtype=torch.float32)
            edges = torch.tensor(new_edges, dtype=torch.long)
            nodes_global_ids = torch.tensor(new_global_ids, dtype=torch.long)
        
        else:
            nodes = torch.tensor(nodes_xy, dtype=torch.float32) if nodes_xy else torch.zeros((0, 2), dtype=torch.float32)
            nodes_global_ids = torch.tensor(nodes_global_ids, dtype=torch.long)
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