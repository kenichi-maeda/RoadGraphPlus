import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .roadgraph_ds import RoadGraphDataset
from .collate import collate

class RoadGraphDataModule(pl.LightningDataModule):
    def __init__(self, root, max_items=20, batch_size=4):
        super().__init__()
        self.root = root
        self.max_items = max_items
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.ds = RoadGraphDataset(self.root, max_items=self.max_items)


    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate
        )