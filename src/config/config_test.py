from unittest import TestCase
import os
from . import ConfigProvider


class ConfigTest(TestCase):

    @classmethod
    def setUpClass(cls):
        script_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_path, 'test_config.yaml')
        cls.config = ConfigProvider.load_config(config_file)

    def test_training_config(self):
        training_config = self.config.get_training_config()

        assert training_config is not None
        assert training_config.batch_size == 32
        assert training_config.epochs == 40000
        assert training_config.learning_rate == 0.1
        assert training_config.lr_decay_rate == 0.1

    def test_test_config(self):
        test_config = self.config.get_test_config()

        assert test_config is not None
        assert test_config.is_recursive is True
        assert test_config.show_plot() is False
        assert test_config.batch_size == 32

    def test_gpu_config(self):
        gpu_config = self.config.get_gpu_config()

        assert gpu_config is not None
        assert gpu_config.memory_fraction == 0.85
        assert gpu_config.is_tf_growth_allowed() is True

    def test_logging_config(self):
        logging_config = self.config.get_logging_config()

        assert logging_config is not None
        assert logging_config.console_logs_active() is True
        assert logging_config.get_file_log_level() == 'DEBUG'
        assert logging_config.get_log_file() == 'crnn.log'
