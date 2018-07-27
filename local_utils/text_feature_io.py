from . import TextFeatureWriter, TextFeatureReader


class TextFeatureIO(object):
    """
        Implement a crnn feature io manager
    """
    def __init__(self):
        self.__writer = TextFeatureWriter()
        self.__reader = TextFeatureReader()
        return

    @property
    def writer(self):
        return self.__writer

    @property
    def reader(self):
        return self.__reader
