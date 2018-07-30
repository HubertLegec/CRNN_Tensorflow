import os.path as ops
import argparse
from config import ConfigProvider, GlobalConfig
from logger import LogFactory
from eval import RecursiveCrnnTester, BasicCrnnTester


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, metavar='PATH', help='Path to config file')
    parser.add_argument('-d', '--dataset_dir', type=str, help='Where you store the test tfrecords data')
    parser.add_argument('-w', '--weights_path', type=str, help='Where you store the shadow net weights')
    return parser.parse_args()


def test_shadownet(dataset_dir: str, weights_path: str, config: GlobalConfig):
    log = LogFactory.get_logger()
    is_recursive = config.get_test_config().is_recursive
    tfrecords_path = ops.join(dataset_dir, 'test_feature.tfrecords')
    tester = RecursiveCrnnTester(tfrecords_path, weights_path, config) if is_recursive else BasicCrnnTester(tfrecords_path, weights_path, config)
    accuracy = tester.run()
    log.info('Mean test accuracy is {:5f}'.format(accuracy))


if __name__ == '__main__':
    params = parse_params()
    config_file = params.config
    config = ConfigProvider.load_config(config_file)
    LogFactory.configure(config.get_logging_config())
    test_shadownet(params.dataset_dir, params.weights_path, config)
