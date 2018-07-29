
class LoggingConfig:
    def __init__(self, yaml_config):
        self._config = yaml_config

    def console_logs_active(self) -> bool:
        return self._config['console_logs']

    def file_logs_active(self) -> bool:
        return self._config['file_logs']

    def get_console_log_level(self) -> str:
        return self._config['console_log_level']

    def get_file_log_level(self) -> str:
        return self._config['file_log_level']

    def get_log_file(self) -> str:
        return self._config['log_file']
