import importlib
import os
import random
import time
import wandb

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.unet import RoadMapUNet
from src.data.unet_dm import UNetDataModule


def main():
    seed_everything(42) 

    dm = UNetDataModule(
        root="/users/kmaeda2/scratch/DeepGrobe/",
        batch_size=16,
        max_items=None
    )

    wandb_logger = WandbLogger(
        project="roadgraph",
        name="unet_training_100",
        log_model=False
    )

    model = RoadMapUNet(
        lr=1e-4, 
        weight_decay=1e-5, 
    )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_unet/",
        save_last=True,
        save_top_k=0
    )

    trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        max_epochs=100,
        accelerator="auto",
        log_every_n_steps=10
    )
    trainer.fit(model, datamodule=dm)

    # wandb.finish()


if __name__ == "__main__":
    main()
