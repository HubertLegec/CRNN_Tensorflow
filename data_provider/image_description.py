import cv2
import numpy as np
from os.path import join, basename


class ImageDescription:
    def __init__(self, path, dataset_dir, label):
        self.image = self._load_image(path, dataset_dir)
        self.label = label
        self.name = basename(path)
        self.encoded_label = None

    def _load_image(self, path, dataset_dir):
        tmp = cv2.imread(join(dataset_dir, path), cv2.IMREAD_COLOR)
        return cv2.resize(tmp, dsize=(100, 32))

    def encode_label(self, map_char):
        self.encoded_label = [map_char(char) for char in self.label]
        return self.encoded_label, len(self.encoded_label)

    def image_as_bytes(self):
        return bytes(list(np.reshape(self.image, [100 * 32 * 3])))
