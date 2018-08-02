from logger import LogFactory


class Lexicon:
    def __init__(self, lexicon_path):
        self._log = LogFactory.get_logger()
        self._lexicon = self._read_file(lexicon_path)

    def get_word_by_index(self, index: int) -> str:
        label = self._lexicon[index]
        self._log.info('Word for index {}: {}'.format(index, label))
        return label

    def _read_file(self, path):
        self._log.info("Reading lexicon file...")
        with open(path, 'r') as lexicon_file:
            lines = [l.strip() for l in lexicon_file.readlines()]
        self._log.info("{} words in dictionary".format(len(lines)))
        return lines
