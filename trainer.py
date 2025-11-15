import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.baseline_model import BaselineModel
from src.data.roadgraph_dm import RoadGraphDataModule


def main():
    dm = RoadGraphDataModule(
        root="/oscar/home/kmaeda2/RoadGraphPlus",
        max_items=500,
        batch_size=16
    )

    model = BaselineModel(lr=1e-4, weight_decay=1e-5)

    trainer = Trainer(
        enable_checkpointing=False,
        enable_progress_bar=True,
        max_epochs=50,
        accelerator="auto",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
