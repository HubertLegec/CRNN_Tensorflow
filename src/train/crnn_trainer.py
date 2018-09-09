import os
import time
import tensorflow as tf
import numpy as np
import os.path as ops
from config import GlobalConfig
from logger import LogFactory
from utils import TextFeatureIO, calculate_array_mean, get_batch_accuracy
from crnn_model import CRNN


class CrnnTrainer:
    def __init__(self, config: GlobalConfig, dataset_dir: str, weights_path: str = None):
        self._log = LogFactory.get_logger()
        self._dataset_dir = dataset_dir
        self._weights_path = weights_path
        self._config = config
        self._display_step = config.get_training_config().display_step
        self._decoder = TextFeatureIO().reader
        self._saver = None
        self._tboard_save_path = 'tboard'
        self._model_save_path = self._get_model_saver_path()

    def train(self):
        training_config = self._config.get_training_config()
        images, labels, imagenames = self._build_data_feed(training_config.batch_size)
        net_out = self._build_net_model(images)
        cost = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_out, sequence_length=25 * np.ones(training_config.batch_size)))
        decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(training_config.batch_size), merge_repeated=False)
        sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
        global_step = tf.Variable(0, name='global_step', trainable=False)
        starter_learning_rate = training_config.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   training_config.lr_decay_steps, training_config.lr_decay_rate,
                                                   staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)
        merge_summary_op = self._configure_tf_summary(cost, learning_rate, sequence_dist)
        self._saver = tf.train.Saver()
        sess = self._create_session()
        summary_writer = tf.summary.FileWriter(self._tboard_save_path)
        summary_writer.add_graph(sess.graph)
        # Set the training parameters
        train_epochs = training_config.epochs

        with sess.as_default():
            self._initialize_model(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for epoch in range(train_epochs):
                self._train_epoch(sess, summary_writer, epoch, optimizer, cost, sequence_dist, decoded, labels, merge_summary_op)
            coord.request_stop()
            coord.join(threads=threads)
        sess.close()
        self._log.info('Training finished.')

    def _train_epoch(self, sess, summary_writer, epoch, optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op):
        _, c, seq_distance, preds_r, gt_labels_r, summary = sess.run(
            [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])
        preds = self._decoder.sparse_tensor_to_str(preds_r[0])
        gt_labels = self._decoder.sparse_tensor_to_str(gt_labels_r)
        accuracy = get_batch_accuracy(preds, gt_labels)
        mean_accuracy = calculate_array_mean(accuracy)
        self._log_epoch_stats(c, epoch, mean_accuracy, seq_distance)
        summary_writer.add_summary(summary=summary, global_step=epoch)
        self._saver.save(sess=sess, save_path=self._model_save_path, global_step=epoch)

    def _log_epoch_stats(self, c, epoch, mean_accuracy, seq_distance):
        if epoch % self._display_step == 0:
            self._log.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(epoch + 1, c, seq_distance, mean_accuracy))

    def _create_session(self):
        gpu_config = self._config.get_gpu_config()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_config.memory_fraction
        sess_config.gpu_options.allow_growth = gpu_config.is_tf_growth_allowed()
        return tf.Session(config=sess_config)

    def _build_data_feed(self, batch_size):
        self._log.info('Build data feed...')
        images, labels, imagenames = self._decoder.read_features(
            ops.join(self._dataset_dir, 'train_feature.tfrecords'), num_epochs=None
        )
        inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
            tensors=[images, labels, imagenames], batch_size=batch_size, capacity=1000 + 2 * 32, min_after_dequeue=100, num_threads=4)
        inputdata = tf.cast(x=inputdata, dtype=tf.float32)
        return inputdata, input_labels, input_imagenames

    def _build_net_model(self, input_data):
        self._log.info('Build net model...')
        crnn = CRNN(phase='Train', hidden_nums=256, seq_length=25, num_classes=37)
        with tf.variable_scope('shadow', reuse=False):
            net_out = crnn.build(inputdata=input_data)
        return net_out

    @classmethod
    def _get_model_saver_path(cls):
        model_save_dir = 'model'
        if not ops.exists(model_save_dir):
            os.makedirs(model_save_dir)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'crnn_{:s}.ckpt'.format(str(train_start_time))
        return ops.join(model_save_dir, model_name)

    def _configure_tf_summary(self, cost, learning_rate, sequence_dist):
        self._log.info('Configure TF summary...')
        if not ops.exists(self._tboard_save_path):
            os.makedirs(self._tboard_save_path)
        tf.summary.scalar(name='Cost', tensor=cost)
        tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
        tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
        return tf.summary.merge_all()

    def _initialize_model(self, sess):
        if self._weights_path is None:
            self._log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            self._log.info('Restore model from {:s}'.format(self._weights_path))
            self._saver.restore(sess=sess, save_path=self._weights_path)
