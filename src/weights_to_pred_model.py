import argparse
import tensorflow as tf
import numpy as np
from os.path import basename, dirname
from tensorflow.python.framework import graph_util, graph_io
from crnn_model import ShadowNet

BATCH_SIZE = 16


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, metavar='PATH', help='Where you store the weights')
    parser.add_argument('-o', '--output', type=str, metavar='PATH', help='Where you save model')
    return parser.parse_args()


def save_graph(sess, output_path):
    output_node_names = "output"
    output_fld = dirname(output_path)
    output_graph = basename(output_path)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_node_names])
    graph_io.write_graph(constant_graph, output_fld, output_graph, as_text=False)


def save_model(weights_path: str, output_path: str):
    inputdata = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 32, 100, 3], name='input')

    net = ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)
    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25 * np.ones(BATCH_SIZE), merge_repeated=False)
    sparse_tensor_values = tf.to_int32(decodes[0]).values
    sparse_tensor_indices = tf.to_int32(decodes[0]).indices
    flattened_indices = tf.to_int32(tf.reshape(sparse_tensor_indices, [-1]))
    output = tf.concat([flattened_indices, sparse_tensor_values], 0, name='output')

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        save_graph(sess, output_path)


if __name__ == '__main__':
    params = parse_params()
    save_model(params.weights, params.output)
