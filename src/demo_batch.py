import argparse
from os import listdir
from os.path import isfile, join
from time import time
import tensorflow as tf
import numpy as np
from utils import TextFeatureIO, load_and_resize_image
from crnn_model import ShadowNet


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, help='Where you store images',
                        default='data/test_images')
    parser.add_argument('-w', '--weights_path', type=str, help='Where you store the weights',
                        default='model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999')
    return parser.parse_args()


def load_images(image_path: str, files_limit: int):
    onlyfiles = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))][:files_limit]
    return np.array([load_and_resize_image(p) for p in onlyfiles]), onlyfiles


def recognize(image_path: str, weights_path: str, files_limit=4):
    decoder = TextFeatureIO().reader
    images, filenames = load_images(image_path, files_limit)
    images = np.squeeze(images)
    padded_images = np.zeros([32, 32, 100, 3])
    padded_images[:images.shape[0], :, :, :] = images
    tf.reset_default_graph()

    inputdata = tf.placeholder(dtype=tf.float32, shape=[32, 32, 100, 3], name='input')

    images_sh = tf.cast(x=inputdata, dtype=tf.float32)

    # build shadownet
    net = ShadowNet(phase='Test', hidden_nums=256, seq_length=25, num_classes=37)
    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)
    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(32), merge_repeated=False)

    # config tf saver
    saver = tf.train.Saver()
    sess = tf.Session()
    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)
        print("Predict...")
        start_time = time()
        predictions = sess.run(decoded, feed_dict={inputdata: padded_images})
        end_time = time()
        print("Prediction time: {}".format(end_time - start_time))
        preds_res = decoder.sparse_tensor_to_str(predictions[0])

        for i, fname in enumerate(filenames):
            print("{}: {}".format(fname, preds_res[i]))


if __name__ == '__main__':
    params = parse_params()
    recognize(params.image_dir, params.weights_path)
