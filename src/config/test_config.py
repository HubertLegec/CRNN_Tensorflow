
class TestConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    def is_recursive(self) -> bool:
        return self._config['is_recursive']

    def show_plot(self) -> bool:
        return self._config['show_plot']
