
class TrainingConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    def get_batch_size(self) -> int:
        return self._config['batch_size']

    def get_epochs(self) -> int:
        return self._config['epochs']

    def get_display_step(self) -> int:
        return self._config['display_step']

    def get_test_display_step(self) -> int:
        return self._config['test_display_step']

    def get_momentum(self) -> float:
        return self._config['momentum']

    def get_learning_rate(self) -> float:
        return self._config['learning_rate']

    def get_validation_batch_size(self) -> int:
        return self._config['val_batch_size']

    def get_lr_decay_steps(self) -> int:
        return self._config['lr_decay_steps']

    def get_lr_decay_rate(self) -> float:
        return self._config['lr_decay_rate']
