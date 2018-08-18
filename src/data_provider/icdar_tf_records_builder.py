import tensorflow as tf
import numpy as np
from tqdm import tqdm
from logger import LogFactory
from os import path
from utils import FeatureIO
from . import ImageDescription


class IcdarTfRecordsBuilder:
    def __init__(self, annotations_file, output_file):
        self._log = LogFactory.get_logger()
        self._annotations_file = annotations_file
        self._output_file = output_file
        self._encoder = FeatureIO()

    def process(self):
        if not path.exists(self._annotations_file):
            raise IOError(f"File {self._annotations_file} doesn't exists")
        annotations_dir = path.dirname(self._annotations_file)
        self._log.info("Reading annotations file...")
        with open(self._annotations_file, 'r') as anno_file:
            infos = np.array([tmp.strip().split(', ') for tmp in anno_file.readlines()])
        self._log.info(f"Number of rows in annotation file: {infos.shape[0]}")

        self._log.info("Processing...")
        skipped = 0
        with tf.python_io.TFRecordWriter(self._output_file) as writer:
            for info in tqdm(infos):
                try:
                    self.process_item(writer, annotations_dir, info)
                except Exception as e:
                    self._log.warn(f"File {info[0]} skipped")
                    skipped += 1
        self._log.info("Skipped count: {}".format(skipped))

    def process_item(self, writer, annotations_dir, info):
        label = info[1][1:-1].lower()
        descr = ImageDescription(info[0], annotations_dir, label)
        descr.encode_label(lambda character: self.char_to_int(character))
        feat_descr = self.create_feature_example(descr)
        writer.write(feat_descr)

    def char_to_int(self, character):
        encoded = self._encoder.char_to_int(character)
        if encoded > 36:
            raise ValueError("Unsupported character")
        return encoded

    @classmethod
    def create_feature_example(cls, img_descr):
        features = tf.train.Features(feature={
            'labels': FeatureIO.int64_feature(img_descr.encoded_label),
            'images': FeatureIO.bytes_feature(img_descr.image_as_bytes()),
            'imagenames': FeatureIO.bytes_feature(img_descr.name)
        })
        example = tf.train.Example(features=features)
        return example.SerializeToString()
