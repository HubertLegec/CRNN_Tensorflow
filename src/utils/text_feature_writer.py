import tensorflow as tf
import os
import os.path as ops
import sys
from . import FeatureIO


class TextFeatureWriter(FeatureIO):
    """
        Implement the CRNN feature writer
    """
    def __init__(self):
        super(TextFeatureWriter, self).__init__()

    def write_features(self, tfrecords_path, labels, images, imagenames):
        assert len(labels) == len(images) == len(imagenames)

        labels, length = self.encode_labels(labels)

        if not ops.exists(ops.split(tfrecords_path)[0]):
            os.makedirs(ops.split(tfrecords_path)[0])

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for index, image in enumerate(images):
                features = tf.train.Features(feature={
                    'labels': self.int64_feature(labels[index]),
                    'images': self.bytes_feature(image),
                    'imagenames': self.bytes_feature(imagenames[index])
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index+1, len(images), imagenames[index]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
