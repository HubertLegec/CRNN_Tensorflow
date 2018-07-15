import argparse
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np
import cv2
from local_utils import data_utils
from crnn_model import crnn_model


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Where you store images',
                        default='data/test_images')
    parser.add_argument('--weights_path', type=str, help='Where you store the weights',
                        default='model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999')

    return parser.parse_args()


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 32))
    return np.expand_dims(image, axis=0).astype(np.float32)


def load_images(image_path, files_limit):
    onlyfiles = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))][:files_limit]
    return np.array([load_image(p) for p in onlyfiles]), onlyfiles


def recognize(image_path, weights_path, files_limit=3):
    decoder = data_utils.TextFeatureIO().reader
    images, filenames = load_images(image_path, files_limit)
    images = np.squeeze(images)
    tf.reset_default_graph()

    inputdata = tf.placeholder(dtype=tf.float32, shape=[files_limit, 32, 100, 3], name='input')

    images_sh = tf.cast(x=inputdata, dtype=tf.float32)

    # build shadownet
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)
    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)
    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(files_limit), merge_repeated=False)

    # config tf saver
    saver = tf.train.Saver()
    sess = tf.Session()
    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)
        print("Predict...")
        predictions = sess.run(decoded, feed_dict={inputdata: images})
        preds_res = decoder.sparse_tensor_to_str(predictions[0])

        for i, pred in enumerate(preds_res):
            print("{}: {}".format(filenames[i], pred))


if __name__ == '__main__':
    # Inti args
    args = init_args()

    # recognize the image
    recognize(args.image_dir, args.weights_path)
