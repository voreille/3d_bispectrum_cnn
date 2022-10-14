"""
Adapation of the code found in https://github.com/danielewworrall/cubenet
to work with tf.keras
TODO: add bias and batchnorm
"""

import sys
import tensorflow as tf

from src.models.cubenet.groups import V_group, T4_group, S4_group


class GroupConv(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 initializer="glorot",
                 drop_sigma=0.0,
                 group="V",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides if type(strides) is not int else 5 * (strides, )
        self.padding = padding.upper()
        if self.padding == 'REFLECT':
            self.pad = kernel_size // 2
        else:
            self.pad = None

        self.initializer = initializer
        self.group = {"V": V_group, "T4": T4_group, "S4": S4_group}[group]()
        self.group_dim = self.group.group_dim
        self.drop_sigma = drop_sigma

    def build(self, input_shape):
        self.x_shape = input_shape
        self.kernel = self.add_weight(name="kernel",
                                      shape=(
                                          self.kernel_size[0],
                                          self.kernel_size[1],
                                          self.kernel_size[2],
                                          input_shape[-1] * input_shape[-2] *
                                          self.filters,
                                      ),
                                      initializer=self.initializer,
                                      trainable=True)

    def get_kernel_n(self):
        kernel_n = self.group.get_Grotations(self.kernel)
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
        batch_size, depth, height, width, channels, _ = tf.shape(x)
        kernel_n = self.get_kernel_n()

        # Gaussian dropout
        if training and self.drop_sigma > 0.0:
            kernel_n *= (
                1 + self.drop_sigma * tf.random.normal(tf.shape(kernel_n)))

        x_n = tf.reshape(
            x, [batch_size, depth, height, width, channels * self.group_dim])

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
        return y
