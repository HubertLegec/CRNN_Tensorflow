import math
import tensorflow as tf
import numpy as np
from tqdm import trange
from time import time
from utils import calculate_array_mean, get_batch_accuracy, batch_levenshtein_distance
from . import CrnnTester


class RecursiveCrnnTester(CrnnTester):
    def load_data(self):
        """
        :return: (tuple) images, labels, imagenames
        """
        images_t, labels_t, imagenames_t = self._decoder.read_features(self._tfrecords_path)
        return tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                              batch_size=self._batch_size,
                              capacity=1000 + 32 * 2,
                              num_threads=4)

    def test(self, decoded, imagenames_sh, images_sh, labels_sh, sess):
        number_of_batches = self._calculate_number_of_batches()
        accuracy = []
        distance = []
        batches = trange(number_of_batches)
        batches.set_description('Processing batches')
        for _ in batches:
            start_time = time()
            predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
            imagenames = self._decode_imagenames(imagenames)
            preds_res = self._decoder.sparse_tensor_to_str(predictions[0])
            gt_res = self._decoder.sparse_tensor_to_str(labels)
            end_time = time()
            self._recognition_time.append(end_time - start_time)
            batch_accuracy = get_batch_accuracy(preds_res, gt_res)
            batch_distance = batch_levenshtein_distance(preds_res, gt_res)
            self._print_result(preds_res, images, gt_res, imagenames)
            accuracy.extend(batch_accuracy)
            distance.extend(batch_distance)
        return calculate_array_mean(accuracy), calculate_array_mean(distance)

    def _print_result(self, predictions, images, labels, imagenames):
        for index, image in enumerate(images):
            self._log.debug(
                'Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(imagenames[index], labels[index], predictions[index])
            )

    def _calculate_number_of_batches(self) -> int:
        test_sample_count = 0
        for _ in tf.python_io.tf_record_iterator(self._tfrecords_path):
            test_sample_count += 1
        self._log.info("Number of records: {}".format(test_sample_count))
        loops_nums = int(math.ceil(test_sample_count / self._batch_size))
        self._log.info("Number of batches: {}". format(loops_nums))
        return loops_nums

    @classmethod
    def _decode_imagenames(cls, imagenames):
        imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
        return [tmp.decode('utf-8') for tmp in imagenames]
