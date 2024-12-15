from omegaconf import OmegaConf
import argparse
import os

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
    cfg = OmegaConf.load(args.config)
    cfg_copy = OmegaConf.create(cfg)
    gap = 0.2
    lambdas = [round(1.0 - gap * i, 2) for i in range(int(1.0 / gap) + 1)]
    lambda_pairs = [(lr, round(1.0 - lr, 2)) for lr in lambdas]
    for lr, lf in lambda_pairs:
        cfg_copy = OmegaConf.create(cfg)
        cfg_copy.train.experiment.params.lambda1 = lr
        cfg_copy.train.experiment.params.lambda2 = lf
        OmegaConf.save(cfg_copy, f"{os.path.basename(args.config)}{int(lr*10)}.yaml")    