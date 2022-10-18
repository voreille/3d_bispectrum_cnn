"""
Adapation of the code found in https://github.com/danielewworrall/cubenet
to work with tf.keras
"""

import sys
import tensorflow as tf

from src.models.cubenet.groups import V_group, T4_group, S4_group_faster


class GroupConv(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 kernel_initializer="glorot_uniform",
                 drop_sigma=0.0,
                 use_bias=True,
                 bias_initializer="zeros",
                 activation="relu",
                 group="V",
                 share_weights=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.share_weights = share_weights
        self.kernel_size = kernel_size if type(kernel_size) is not int else (
            kernel_size, kernel_size, kernel_size)
        self.strides = strides if type(strides) is not int else 5 * (strides, )
        self.padding = padding.upper()
        if self.padding == 'REFLECT':
            self.pad = kernel_size // 2
        else:
            self.pad = None

        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=0.2, )
        self.group = {
            "V": V_group,
            "T4": T4_group,
            "S4": S4_group_faster
        }[group]()
        self.group_dim = self.group.group_dim
        self.drop_sigma = drop_sigma
        if use_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=(self.filters, ),
                                        initializer=bias_initializer)
        else:
            self.bias = None
        self.activation = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        self.x_shape = input_shape
        fan_in = input_shape[-1] * self.kernel_size[0] * self.kernel_size[
            1] * self.kernel_size[2] * self.group_dim
        fan_out = self.filters * self.kernel_size[0] * self.kernel_size[
            1] * self.kernel_size[2] * self.group_dim
        limit = tf.sqrt(6 / (fan_in + fan_out))
        if self.share_weights:
            group_dim = 1
        else:
            group_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
                input_shape[-2],
                self.filters,
                group_dim,
            ),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-limit,
                maxval=limit,
            ),
            trainable=True)

    def get_kernel_n(self):
        if self.share_weights:
            kernel_n = tf.repeat(self.kernel, self.x_shape[-1], axis=5)
        else:
            kernel_n = self.kernel
        kernel_n = self.group.get_Grotations(
            tf.reshape(kernel_n, [
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.x_shape[-1] * self.filters * self.x_shape[-2]
            ]))
        kernel_n = tf.stack(kernel_n, axis=-1)
        if self.x_shape[-1] == 1:
            kernel_n = tf.reshape(kernel_n, [
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.x_shape[4], -1
            ])
        elif self.x_shape[-1] == self.group_dim:
            kernel_n = tf.reshape(kernel_n, [
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.x_shape[4], self.group_dim, self.filters, self.group_dim
            ])

            kernel_n = tf.stack(self.group.G_permutation(kernel_n), axis=-1)
            kernel_n = tf.reshape(kernel_n, [
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.x_shape[4] * self.group_dim, self.filters * self.group_dim
            ])

        return kernel_n

    def call(self, x, training=None):
        _, depth, height, width, channels, group_dim = x.get_shape().as_list()
        batch_size = tf.shape(x)[0]
        kernel_n = self.get_kernel_n()

        # Gaussian dropout
        if training and self.drop_sigma > 0.0:
            kernel_n *= (
                1 + self.drop_sigma * tf.random.normal(tf.shape(kernel_n)))

        x_n = tf.reshape(
            x, [batch_size, depth, height, width, channels * group_dim])

        if self.pad:
            padding = 'VALID'
            x_n = tf.pad(x_n,
                         [[0, 0], [self.pad, self.pad], [self.pad, self.pad],
                          [self.pad, self.pad], [0, 0]],
                         mode='REFLECT')
        else:
            padding = self.padding

        y = tf.nn.conv3d(x_n, kernel_n, self.strides, padding)
        ysh = y.get_shape().as_list()
        y = tf.reshape(
            y,
            [batch_size, ysh[1], ysh[2], ysh[3], self.filters, self.group_dim])
        if self.bias is not None:
            # broadcast and add bias
            y += tf.reshape(self.bias, [1, 1, 1, 1, self.filters, 1])
        return self.activation(y)


class MaxPoolG(tf.keras.layers.Layer):

    def __init__(self,
                 pool_size=(2, 2, 2),
                 strides=None,
                 padding="valid",
                 data_format=None,
                 name=None,
                 **kwargs):
        super().__init__(name=name)
        self.max_pool = tf.keras.layers.MaxPool3D(pool_size=pool_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  data_format=data_format,
                                                  **kwargs)

    def call(self, x):
        _, depth, height, width, channels, group_dim = x.get_shape().as_list()
        batch_size = tf.shape(x)[0]
        x = tf.reshape(
            x, [batch_size, depth, height, width, channels * group_dim])
        x = self.max_pool(x)
        _, depth, height, width, _ = x.get_shape().as_list()
        x = tf.reshape(x,
                       [batch_size, depth, height, width, channels, group_dim])
        return x


class UpsamplingG(tf.keras.layers.Layer):

    def __init__(self, size=(2, 2, 2), data_format=None, name=None):
        super().__init__(name=name)
        self.upsampling = tf.keras.layers.UpSampling3D(size=size,
                                                       data_format=data_format)

    def call(self, x):
        _, depth, height, width, channels, group_dim = x.get_shape().as_list()
        batch_size = tf.shape(x)[0]
        x = tf.reshape(
            x,
            [batch_size, depth, height, width, channels * group_dim],
        )
        x = self.upsampling(x)
        _, depth, height, width, _ = x.get_shape().as_list()
        x = tf.reshape(
            x,
            [batch_size, depth, height, width, channels, group_dim],
        )
        return x
