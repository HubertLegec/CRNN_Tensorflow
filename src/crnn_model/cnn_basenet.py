import tensorflow as tf
from abc import ABC


class CNNBaseModel(ABC):
    """
    Base model for other specific cnn ctpn_models.
    The base convolution neural networks mainly implement some useful cnn functions
    """

    @staticmethod
    def conv2d(inputdata, out_channel: int, kernel_size: int, padding='SAME', stride=1, w_init=None, nl=tf.identity, name=None):
        """
        Packing the TensorFlow conv2d function.

        Arguments:
            :param name: op name
            :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other unknown dimensions.
            :param out_channel: number of output channel.
            :param kernel_size: int so only support square kernel convolution
            :param padding: 'VALID' or 'SAME'
            :param stride: int so only support square stride
            :param w_init: initializer for convolution weights
            :param nl: a tensorflow identify function

        Returns:
            :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            filter_shape = [kernel_size, kernel_size] + [in_channel, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1]
            else:
                strides = [1, stride, stride, 1]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)

            conv = tf.nn.conv2d(inputdata, w, strides, padding)
            ret = nl(conv, name=name)
        return ret

    @staticmethod
    def separable_conv2d(inputdata, out_channel: int, kernel_size: int, padding='SAME', stride=1, w_init=None, nl=tf.identity, name=None):
        """
        Packing the TensorFlow separable_conv2d function.

        Arguments:
            :param name: op name
            :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other unknown dimensions.
            :param out_channel: number of output channel.
            :param kernel_size: int so only support square kernel convolution
            :param padding: 'VALID' or 'SAME'
            :param stride: int so only support square stride
            :param w_init: initializer for convolution weights
            :param nl: a tensorflow identify function

        Returns:
            :return: tf.Tensor named ``output``
        """
        channel_multiplier = 4
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1]
            else:
                strides = [1, stride, stride, 1]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()

            w1 = tf.get_variable('W1', [kernel_size, kernel_size, in_channel, channel_multiplier], initializer=w_init)
            w2 = tf.get_variable('W2', [1, 1, channel_multiplier * in_channel, out_channel], initializer=w_init)

            conv = tf.nn.separable_conv2d(inputdata, w1, w2, strides, padding)
            ret = nl(conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        padding = padding.upper()
        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, noise_shape=None, name=None):
        return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

    @staticmethod
    def layerbn(inputdata, is_training):
        output = tf.contrib.layers.batch_norm(inputdata, scale=True, is_training=is_training, updates_collections=None)
        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        return tf.squeeze(input=inputdata, axis=axis, name=name)
