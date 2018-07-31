import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from . import CrnnTester
from config import GlobalConfig
from utils import calculate_mean_accuracy, get_batch_accuracy


class BasicCrnnTester(CrnnTester):
    def __init__(self, tfrecords_path: str, weights_path: str, config: GlobalConfig):
        super().__init__(tfrecords_path, weights_path, config)
        self._show_plot = config.get_test_config().show_plot()

    def load_data(self):
        images_t, labels_t, imagenames_t = self._decoder.read_features(self._tfrecords_path, num_epochs=None)
        return tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                      batch_size=self._batch_size,
                                      capacity=1000 + 32 * 2,
                                      min_after_dequeue=2,
                                      num_threads=4)

    def test(self, decoded, imagenames_sh, images_sh, labels_sh, sess):
        start_time = time()
        predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
        imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
        imagenames = [tmp.decode('utf-8') for tmp in imagenames]
        preds_res = self._decoder.sparse_tensor_to_str(predictions[0])
        gt_res = self._decoder.sparse_tensor_to_str(labels)
        end_time = time()
        self._recognition_time.append(end_time - start_time)
        accuracy = get_batch_accuracy(preds_res, gt_res)
        for index, image in enumerate(images):
            self._log.info(
                'Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(imagenames[index], gt_res[index], preds_res[index])
            )
            if self._show_plot:
                plt.imshow(image[:, :, (2, 1, 0)])
                plt.show()
        return calculate_mean_accuracy(accuracy)
