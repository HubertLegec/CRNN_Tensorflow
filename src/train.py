import os.path as ops
import argparse
from config import ConfigProvider
from logger import LogFactory
from train import CrnnTrainer


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, metavar='PATH', help='Path to config file')
    parser.add_argument('-d', '--dataset_dir', type=str, metavar='PATH', help='Where you store the dataset')
    parser.add_argument('-w', '--weights_path', type=str, metavar='PATH', help='Where you store the pretrained weights')
    return parser.parse_args()


if __name__ == '__main__':
    params = parse_params()
    dataset_dir = params.dataset_dir
    config_file = params.config
    config = ConfigProvider.load_config(config_file)
    LogFactory.configure(config.get_logging_config())
    if not ops.exists(dataset_dir):
        raise ValueError("{:s} doesn't exist".format(dataset_dir))
    trainer = CrnnTrainer(config, dataset_dir, params.weights_path)
    trainer.train()
