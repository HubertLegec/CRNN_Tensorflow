import argparse
import cv2
import tensorflow as tf
import numpy as np
import os.path as osp
from tensorflow.python.platform import gfile

"""
Script to test predictions on frozen TF graph
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, metavar='PATH')
    parser.add_argument('-m', '--model', type=str, metavar='PATH')
    return parser.parse_args()


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 32))
    return np.expand_dims(image, axis=0).astype(np.float32)


if __name__ == '__main__':
    args = parse_args()
    model_file = args.model  # model/crnn_freeze.pb
    if not osp.exists('./model/crnn_freeze.pb'):
        raise ValueError('{:s} doesn\'t exist'.format(args.image_path))
    image = load_image(args.image_path)
    tf.reset_default_graph()
    with tf.Session(graph=tf.Graph()) as sess:
        with gfile.FastGFile('./model/crnn_freeze.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], './SavedModel/')
        inputdata = tf.get_default_graph().get_tensor_by_name("input:0")
        output = tf.get_default_graph().get_tensor_by_name("output:0")
        preds = sess.run(output, feed_dict={inputdata: image})
        print(f"Predictions:\n{preds[0]}")
