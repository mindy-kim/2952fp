import torch
from omegaconf import OmegaConf
import argparse
import datetime
import os
from utils.utils import instantiate_from_config

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from models.transformer import Transformer

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

class SaveInputCallback(Callback):
    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = out_dir
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Save the data for each batch
        save_path = os.path.join(self.out_dir, f"batch_{batch_idx}.pt")
        torch.save(batch, save_path)


if __name__ == '__main__':
    args = get_parser()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    cfg = OmegaConf.load(args.config)
    cfg.data.params['Nx'], cfg.data.params['Ny'] = cfg.model.params.dim_x, cfg.model.params.dim_y

    data = instantiate_from_config(cfg.data)
    
    cfg_fname = os.path.splitext(os.path.split(args.config)[-1])[0]
    nowname = now + "_" + cfg_fname
    logdir = os.path.join(args.logdir, nowname)
    os.makedirs(logdir, exist_ok=True)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    os.makedirs(ckptdir, exist_ok=True)

    inputdir = os.path.join(logdir, "inputs")
    os.makedirs(inputdir, exist_ok=True)

    model : Transformer = instantiate_from_config(cfg.model)
    if 'ckpt_path' in cfg.model:
        model = Transformer.load_from_checkpoint(cfg.model.ckpt_path, **cfg.model.params)
        print(cfg.train, cfg.train.experiment)
        if 'experiment' in  cfg.train and cfg.train.experiment.name == "unlearning":
            model.set_lambda(cfg.train.experiment.params.lambda1, 
                             cfg.train.experiment.params.lambda2)

    if cfg.train:
        bs, base_lr = cfg.train.batch_sz, cfg.train.lr
        model.learning_rate = base_lr
        dataloader = DataLoader(data, batch_size=cfg.train.batch_sz)
        lightning_cfg = cfg.train.lightning_cfg if 'lightning_cfg' in cfg.train else {}
        
        trainer = pl.Trainer(
            logger=TensorBoardLogger(save_dir=logdir),
            callbacks=[
                ModelCheckpoint(
                    dirpath=os.path.join(ckptdir, 'trainstep_checkpoints'),
                    verbose=True,
                    every_n_epochs=5,
                    save_weights_only=True
                )
            ],
            **lightning_cfg
        )

        trainer.fit(model, train_dataloaders=dataloader)

    if cfg.hijack:
        # train input to hijack model and produce bad (what we want unlearned) output
        # train model to unlearn bad tasks
        # see how model performs on hijacked data?

        # train model + save training data
        bs, base_lr = cfg.train.batch_sz, cfg.train.lr
        model.learning_rate = base_lr
        dataloader = DataLoader(data, batch_size=cfg.train.batch_sz)
        lightning_cfg = cfg.train.lightning_cfg if 'lightning_cfg' in cfg.train else {}
        
        trainer = pl.Trainer(
            logger=TensorBoardLogger(save_dir=logdir),
            callbacks=[
                ModelCheckpoint(
                    dirpath=os.path.join(ckptdir, 'trainstep_checkpoints'),
                    verbose=True,
                    every_n_epochs=5,
                    save_weights_only=True
                ),
                SaveInputCallback(out_dir=inputdir)
            ],
            **lightning_cfg
        )

        trainer.fit(model, train_dataloaders=dataloader)

        # 

        for param in model.parameters():
            param.requires_grad = False
        
        input.requires_grad = True




    
