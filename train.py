import torch
from omegaconf import OmegaConf
import argparse

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

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    print(args.config)
    cfg = OmegaConf.load(args.config)
    data = MULData(**cfg.data)
