import yaml
from . import GlobalConfig


class ConfigProvider:
    config = None

    @classmethod
    def load_config(cls, path: str) -> GlobalConfig:
        """
        Loads configuration from file
        :param path: path to config file
        :return: config object
        """
        with open(path, "r") as config_file:
            config_dict = yaml.load(config_file)
            cls.config = GlobalConfig(config_dict)
        return cls.config

    @classmethod
    def get_config(cls) -> GlobalConfig:
        if cls.config is None:
            raise ValueError("Config is not loaded")
        return cls.config
