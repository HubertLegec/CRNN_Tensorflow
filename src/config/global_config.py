from . import TrainingConfig, GpuConfig, TestConfig


class GlobalConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    def get_training_config(self) -> TrainingConfig:
        return TrainingConfig(self._config['training'])

    def get_gpu_config(self) -> GpuConfig:
        return GpuConfig(self._config['gpu'])

    def get_test_config(self) -> TestConfig:
        return TestConfig(self._config['test'])
