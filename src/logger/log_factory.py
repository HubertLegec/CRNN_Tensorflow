import logging
from config import LoggingConfig


class LogFactory:
    ROOT_LOGGER_NAME = 'CRNN'
    FORMATTER = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s: %(message)s')
    DEFAULT_LOG_LEVEL = 'INFO'
    DEFAULT_LOG_FILE = 'crnn.log'
    log = None

    @classmethod
    def configure(cls, config: LoggingConfig):
        print("Configuring logger...")
        cls.log = logging.getLogger(cls.ROOT_LOGGER_NAME)
        cls.log.setLevel(logging.DEBUG)
        fh = cls._get_file_logger(config.file_logs_active(), config.get_file_log_level(), config.get_log_file())
        ch = cls._get_console_logger(config.console_logs_active(), config.get_console_log_level())
        if fh:
            cls.log.addHandler(fh)
        if ch:
            cls.log.addHandler(ch)

    @classmethod
    def get_logger(cls):
        if cls.log:
            return cls.log
        print("!!! Logger is not configured !!!")
        cls.log = logging.getLogger(cls.ROOT_LOGGER_NAME)
        cls.log.setLevel(logging.DEBUG)
        return cls.log

    @classmethod
    def _get_file_logger(cls, file_logs, log_level, log_file):
        if not file_logs:
            return None
        log = logging.FileHandler(log_file)
        log.setLevel(log_level)
        log.setFormatter(cls.FORMATTER)
        return log

    @classmethod
    def _get_console_logger(cls, console_logs, log_level):
        if not console_logs:
            return None
        log = logging.StreamHandler()
        log.setLevel(log_level)
        log.setFormatter(cls.FORMATTER)
        return log
