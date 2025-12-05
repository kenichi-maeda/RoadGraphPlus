import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.baseline_model import BaselineModel
from src.models.high_resolution import HighResolutionModel  
from src.data.roadgraph_dm import RoadGraphDataModule


def main():
    seed_everything(42) 

    dm = RoadGraphDataModule(
        root="/users/clingzhi/RoadGraphPlus", 
        batch_size=32,
        max_items=None
    )

    model = HighResolutionModel.load_from_checkpoint(  
        "checkpoints/last_G_32.ckpt",
        warmup_epochs=10,
        anneal_epochs=20,
        min_gt_prob=0.2,
        lr=1e-3,
        weight_decay=1e-5
    )
    model.eval()  

    trainer = Trainer(
        accelerator="auto",
        devices=1,  
        log_every_n_steps=1,
    )

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()