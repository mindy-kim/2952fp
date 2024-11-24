import torch
from omegaconf import OmegaConf
import argparse
import datetime
import os
from utils.utils import instantiate_from_config

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from data import MULData

def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        const=True,
        required=True,
        nargs='?',
        help="path to experiment config",
    )

    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        const=True,
        default='./logs',
        nargs='?',
        help="path to experiment config",
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    cfg = OmegaConf.load(args.config)
    data = MULData(**cfg.data)
    
    cfg_fname = os.path.splitext(os.path.split(args.config)[-1])[0]
    print(os.path.split(args.config[0])[-1])
    nowname = now + "_" + cfg_fname
    logdir = os.path.join(args.logdir, nowname)
    os.makedirs(logdir, exist_ok=True)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    os.makedirs(ckptdir, exist_ok=True)

    model = instantiate_from_config(cfg.model)

    if cfg.train:
        bs, base_lr = cfg.train.batch_sz, cfg.train.lr
        model.learning_rate = base_lr
        dataloader = DataLoader(data, batch_size=bs, shuffle=True)
        
        trainer = pl.Trainer(
            logger=TensorBoardLogger(save_dir=logdir),
            max_epochs=cfg.train.epochs,
            callbacks=[
                ModelCheckpoint(
                    dirpath=os.path.join(ckptdir, 'trainstep_checkpoints'),
                    verbose=True,
                    every_n_epochs=5,
                    save_weights_only=True
                )
            ]
        )

        trainer.fit(model, train_dataloaders=dataloader)



    