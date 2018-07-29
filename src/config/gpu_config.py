
class GpuConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    def get_memory_fraction(self) -> float:
        return self._config['memory_fraction']

    def is_tf_growth_allowed(self) -> bool:
        return self._config['tf_allow_growth']
