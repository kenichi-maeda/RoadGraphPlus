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
    seed_everything(42) 

    dm = RoadGraphDataModule(
        root="/users/clingzhi/RoadGraphPlus",
        batch_size=32,
        max_items=None
    )

    # Step 3: Evaluation
    model = BaselineModel.load_from_checkpoint(
        "checkpoints/last_G_32.ckpt",  
        warmup_epochs=15,
        anneal_epochs=25,
        min_gt_prob=0.2,
        lr=1e-4,
        weight_decay=1e-4
    )


    trainer = Trainer(
        accelerator="auto",
        log_every_n_steps=1,
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
