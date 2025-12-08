import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .roadgraph_ds import RoadGraphDataset
from .collate import collate
from pathlib import Path
import random

class RoadGraphDataModule_Step3(pl.LightningDataModule):
    def __init__(self, root, max_items=20, batch_size=4, val_frac=0.1, test_frac=0.1):
        super().__init__()
        self.root = Path(root)
        self.max_items = max_items
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac

    def setup(self, stage=None):
        all_files = sorted((self.root / "data/labels").glob("*.json"))
        random.seed(42)
        random.shuffle(all_files)

        training_files = sorted((self.root / "data_augmented/labels").glob("*.json"))

        if self.max_items is not None:
            all_files = all_files[:self.max_items]

        N = len(all_files)
        Nv = int(N * self.val_frac)
        Nt = int(N * self.test_frac)

        self.test_files = all_files[:Nt]
        self.val_files = all_files[Nt: Nt + Nv]
        self.train_files = training_files

        self.train_ds = RoadGraphDataset(self.train_files, augment=True, add_vitrutal=False)
        self.val_ds = RoadGraphDataset(self.val_files, augment=False)
        self.test_ds = RoadGraphDataset(self.test_files, augment=False)

        print("train files", len(self.train_ds))
        print("val files", len(self.val_ds))
        print("test files", len(self.test_ds))


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate
        )