import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.high_resolution import HighResolutionModel
from src.data.roadgraph_dm import RoadGraphDataModule

# Trainer for High Resolution Model, hence the name trainer'H'.py
def main():
    seed_everything(42) 
    
    # Stage 1: Pretraining junctions and offsets
    dm = RoadGraphDataModule(
        root="/oscar/home/dbchanin/RoadGraphPlus",
        batch_size=32,
        max_items=None
    )

    wandb_logger = WandbLogger(
        project="roadgraph",
        name="Step1_High",
        log_model=False
    )

    model = HighResolutionModel(lr=1e-4, 
                          weight_decay=1e-5, 
                          warmup_epochs=50,
                          anneal_epochs=0,
                          lambda_e=0.0,
                          lambda_j=1.0,
                          lambda_o=1.0,
                          pretrain=True,
                          posttrain=False
                          )
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_stage1/",
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

    # Stage 2
    # dm = RoadGraphDataModule(
    #     root="/oscar/home/dbchanin/RoadGraphPlus",
    #     batch_size=32,
    #     max_items=None
    # )

    # wandb_logger = WandbLogger(
    #     project="roadgraph",
    #     name="Step2_High_Geo",
    #     log_model=False
    # )

    # model = HighResolutionModel.load_from_checkpoint(
    #                       "checkpoints_stage1/last_B32.ckpt",
    #                       lr=1e-4, 
    #                       weight_decay=1e-5, 
    #                       warmup_epochs=10,
    #                       anneal_epochs=20,
    #                       posttrain=True,
    #                       lambda_e=1.0,
    #                       lambda_j=1.0,
    #                       lambda_o=1.0)
    
    # checkpoint_cb = ModelCheckpoint(
    #     dirpath="checkpoints_stage2/",
    #     save_last=True,
    #     save_top_k=0
    # )

    # trainer = Trainer(
    #     logger=wandb_logger,
    #     enable_checkpointing=True,
    #     callbacks=[checkpoint_cb],
    #     enable_progress_bar=True,
    #     max_epochs=120,
    #     accelerator="auto",
    #     log_every_n_steps=1,
    #     strategy="ddp_find_unused_parameters_true",
    # )
    # trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
