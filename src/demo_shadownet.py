import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from crnn_model import ShadowNet
from utils import TextFeatureIO, load_and_resize_image
from config import ConfigProvider, GlobalConfig
from logger import LogFactory


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, metavar='PATH', help='Path to config file')
    parser.add_argument('-i', '--image_path', type=str, help='Path to image file')
    parser.add_argument('-w', '--weights_path', type=str, help='Where you store the weights')
    return parser.parse_args()


def recognize(image_path: str, weights_path: str, config: GlobalConfig, is_vis=True):
    logger = LogFactory.get_logger()
    image = load_and_resize_image(image_path)

    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')

    net = ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)

    decoder = TextFeatureIO()

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.get_gpu_config().get_memory_fraction()
    sess_config.gpu_options.allow_growth = config.get_gpu_config().is_tf_growth_allowed()

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        preds = sess.run(decodes, feed_dict={inputdata: image})
        preds = decoder.writer.sparse_tensor_to_str(preds[0])
        logger.info('Predict image {:s} label {:s}'.format(ops.split(image_path)[1], preds[0]))

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            plt.show()

        sess.close()


if __name__ == '__main__':
    params = parse_params()
    image_path = params.image_path
    weights_path = params.weights_path
    config_file = params.config
    config = ConfigProvider.load_config(config_file)
    LogFactory.configure(config.get_logging_config())
    if not ops.exists(image_path):
        raise ValueError("{} doesn't exist".format(image_path))
    recognize(image_path, weights_path, config)
