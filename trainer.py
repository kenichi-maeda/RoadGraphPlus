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
    
    # Stage 1
    # dm = RoadGraphDataModule(
    #     root="/oscar/home/kmaeda2/RoadGraphPlus",
    #     batch_size=32,
    #     max_items=None
    # )

    # wandb_logger = WandbLogger(
    #     project="roadgraph",
    #     name="exp_step1",
    #     log_model=False
    # )

    # model = BaselineModel(lr=1e-3, 
    #                       weight_decay=1e-5, 
    #                       warmup_epochs=50,
    #                       anneal_epochs=1000,
    #                       lambda_e=0.0)
    # checkpoint_cb = ModelCheckpoint(
    #     dirpath="checkpoints_stage1/",
    #     save_last=True,
    #     save_top_k=0
    # )
    # trainer = Trainer(
    #     logger=wandb_logger,
    #     enable_checkpointing=True,
    #     callbacks=[checkpoint_cb],
    #     enable_progress_bar=True,
    #     max_epochs=50,
    #     accelerator="auto",
    #     log_every_n_steps=1,
    #     strategy="ddp_find_unused_parameters_true",
    # )
    # trainer.fit(model, datamodule=dm)

    # Stage 2
    dm = RoadGraphDataModule(
        root="/oscar/home/kmaeda2/RoadGraphPlus",
        batch_size=32,
        max_items=None
    )

    wandb_logger = WandbLogger(
        project="roadgraph",
        name="exp_step2",
        log_model=False
    )

    model = BaselineModel.load_from_checkpoint(
                          "checkpoints_stage1/last.ckpt",
                          lr=1e-3, 
                          weight_decay=1e-5, 
                          warmup_epochs=10,
                          anneal_epochs=20,
                          min_gt_prob=0.2)
    
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
        max_epochs=50,
        accelerator="auto",
        log_every_n_steps=1,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
