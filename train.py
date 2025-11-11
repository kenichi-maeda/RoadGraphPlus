import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import parse_args_main
import datasets

def get_model_modules(args):
    """
    Choose model
    """
    if args.experiment == "baseline":
        module = importlib.import_module("models.baseline_model")
        ModelClass = getattr(module, "BaseLineModel")
    else:
        ...
    return ModelClass(**vars(args))

def main(args):
    if args.wandb:
        logger = WandbLogger(...)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(...)

    trainer = Trainer(
        callbacks=checkpoint_callback,
        check_val_every_n_epoch=...,
        deterministic=...,
        enable_progress_bar=...,
        limit_train_batches=...,
        limit_val_batches=...,
        log_every_n_steps=...,
        logger=logger,
        max_epochs=...,
        num_sanity_val_steps=...,
        profiler=None
    )

    dm = ...

    model = get_model_modules(args)

    trainer.fit(model, datamodule=dm)

