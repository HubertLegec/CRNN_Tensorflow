import argparse
import tensorflow as tf
import numpy as np
from time import time
from os.path import exists
from tensorflow.python.platform import gfile
from utils import load_and_resize_image

"""
Script to test predictions on frozen TF graph
"""

BATCH_SIZE = 16


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, metavar='PATH')
    parser.add_argument('-m', '--model', type=str, metavar='PATH')
    return parser.parse_args()


if __name__ == '__main__':
    params = parse_params()
    model_file = params.model
    image_path = params.image_path
    if not exists(model_file):
        raise ValueError('{:s} doesn\'t exist'.format(model_file))
    image = load_and_resize_image(image_path)
    padded_images = np.zeros([BATCH_SIZE, 32, 100, 3])
    padded_images[:image.shape[0], :, :, :] = image
    tf.reset_default_graph()
    with tf.Session(graph=tf.Graph()) as sess:
        with gfile.FastGFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        inputdata = tf.get_default_graph().get_tensor_by_name("input:0")
        output = tf.get_default_graph().get_tensor_by_name("output:0")
        start_time = time()
        preds = sess.run(output, feed_dict={inputdata: padded_images})
        end_time = time()
        print("Recognition time: {}".format(end_time - start_time))
        print(f"Predictions:\n{preds[0]}")
