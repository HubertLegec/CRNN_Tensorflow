import tensorflow as tf
import numpy as np
from abc import ABC


class CNNBaseModel(ABC):
    """
    Base model for other specific cnn ctpn_models.
    The base convolution neural networks mainly implement some useful cnn functions
    """

    @staticmethod
    def conv2d(inputdata, out_channel: int, kernel_size: int, padding='SAME', stride=1, w_init=None, nl=tf.identity, name=None):
        """
        Packing the tensorflow conv2d function.

        Arguments:
            :param name: op name
            :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other unknown dimensions.
            :param out_channel: number of output channel.
            :param kernel_size: int so only support square kernel convolution
            :param padding: 'VALID' or 'SAME'
            :param stride: int so only support square stride #FIXME fix comment
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
        Packing the tensorflow separable_conv2d function.

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

            w1 = tf.get_variable('W', filter_shape[:2], initializer=w_init)
            w2 = tf.get_variable('W', filter_shape[:2], initializer=w_init)

            conv = tf.nn.separable_conv2d(inputdata, w1, w2, strides, padding)
            ret = nl(conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        return tf.nn.sigmoid(x=inputdata, name=name)

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
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
        if stride is None:
            stride = kernel_size
        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]
        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]
        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']
        axis = [1, 2] if data_format == 'NHWC' else [2, 3]
        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True, data_format='NHWC', name=None):
        """
        Arguments:
            :param name:
            :param inputdata:
            :param epsilon: epsilon to avoid divide-by-zero.
            :param use_bias: whether to use the extra affine transformation or not.
            :param use_scale: whether to use the extra affine transformation or not.
            :param data_format:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, noise_shape=None, name=None):
        return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None, nl=tf.identity, use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        Arguments:
            :param inputdata:  a tensor to be flattened except for the first dimension.
            :param out_dim: output dimension
            :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
            :param b_init: initializer for b. Defaults to zero
            :param nl: a nonlinearity function
            :param use_bias: whether to use bias.
            :param name:

        Returns:
            :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: nl(x, name='output'), use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init, trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training):
        output = tf.contrib.layers.batch_norm(inputdata, scale=True, is_training=is_training, updates_collections=None)
        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        return tf.squeeze(input=inputdata, axis=axis, name=name)
