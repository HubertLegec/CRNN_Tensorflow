
class TrainingConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    @property
    def batch_size(self) -> int:
        return self._config['batch_size']

    @property
    def epochs(self) -> int:
        return self._config['epochs']

    @property
    def display_step(self) -> int:
        return self._config['display_step']

    @property
    def test_display_step(self) -> int:
        return self._config['test_display_step']

    @property
    def momentum(self) -> float:
        return self._config['momentum']

    @property
    def learning_rate(self) -> float:
        return self._config['learning_rate']

    @property
    def validation_batch_size(self) -> int:
        return self._config['val_batch_size']

    @property
    def lr_decay_steps(self) -> int:
        return self._config['lr_decay_steps']

    @property
    def lr_decay_rate(self) -> float:
        return self._config['lr_decay_rate']
