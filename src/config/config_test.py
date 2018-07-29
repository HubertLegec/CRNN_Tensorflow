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
        assert training_config.get_batch_size() == 32
        assert training_config.get_epochs() == 40000
        assert training_config.get_learning_rate() == 0.1
        assert training_config.get_lr_decay_rate() == 0.1

    def test_test_config(self):
        test_config = self.config.get_test_config()

        assert test_config is not None
        assert test_config.is_recursive() is True
        assert test_config.show_plot() is False

    def test_gpu_config(self):
        gpu_config = self.config.get_gpu_config()

        assert gpu_config is not None
        assert gpu_config.get_memory_fraction() == 0.85
        assert gpu_config.is_tf_growth_allowed() is True
