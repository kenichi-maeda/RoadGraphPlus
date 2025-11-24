import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.segbased_model import SegBasedModel
from src.data.roadgraph_dm import RoadGraphDataModule


def main():
    seed_everything(42) 
    
    dm = RoadGraphDataModule(
        root="/oscar/home/kmaeda2/RoadGraphPlus",
        batch_size=32,
        max_items=None
    )

    wandb_logger = WandbLogger(
        project="roadgraph",
        name="exp_with_unet",
        log_model=False
    )

    model = SegBasedModel(
                          unet_path="checkpoints_unet/last.ckpt",
                          lr=1e-4, 
                          weight_decay=1e-5, 
                          warmup_epochs=10,
                          anneal_epochs=20,
                          min_gt_prob=0.2,
                          )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_stage2/",
        save_last=True,
        save_top_k=0
    )

    trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        max_epochs=40,
        accelerator="auto",
        log_every_n_steps=1,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
