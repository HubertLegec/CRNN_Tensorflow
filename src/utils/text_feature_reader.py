import tensorflow as tf
import os.path as ops
from . import FeatureIO


class TextFeatureReader(FeatureIO):
    """
        Implement the CRNN feature reader.
    """
    def __init__(self):
        super(TextFeatureReader, self).__init__()

    @staticmethod
    def read_features(tfrecords_path: str, num_epochs: int):
        assert ops.exists(tfrecords_path)

        filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'images': tf.FixedLenFeature((), tf.string),
                                               'imagenames': tf.FixedLenFeature([1], tf.string),
                                               'labels': tf.VarLenFeature(tf.int64),
                                           })
        image = tf.decode_raw(features['images'], tf.uint8)
        images = tf.reshape(image, [32, 100, 3])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        imagenames = features['imagenames']
        return images, labels, imagenames
