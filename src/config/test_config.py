
class TestConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    @property
    def is_recursive(self) -> bool:
        return self._config['is_recursive']

    def show_plot(self) -> bool:
        return self._config['show_plot']

    @property
    def batch_size(self) -> int:
        return self._config['batch_size']
