from pathlib import Path

import tensorflow as tf
import yaml

from src.models.layers_faster import BSHConv3D, SSHConv3D
from src.models.losses import dice_loss, dice_coefficient_hard
from src.models.cubenet.layers import GroupConv


class ResidualLayer3DBase(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size,
        use_batch_norm=True,
        n_conv=1,
        padding='SAME',
        **kwargs,
    ):
        super().__init__(name=kwargs.get("name", None))
        self.kwargs = kwargs
        self.filters = filters
        self.n_conv = n_conv
        self.use_batch_norm = use_batch_norm
        self.padding = padding.upper()
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (
            kernel_size, kernel_size, kernel_size)
        self.activation = tf.keras.layers.Activation(
            kwargs.get("activation", "relu"))
        self.conv = self._get_conv_block()
        self.residual_conv = None

    def _get_conv_block(self):
        raise NotImplementedError()

    def _get_residual_conv_block(self):
        raise NotImplementedError()

    def build(self, input_shape):
        self.c_in = input_shape[-1]
        if input_shape[-1] != self.filters:
            self.residual_conv = self._get_residual_conv_block()

    def call(self, x, training=None):
        residual = self.residual_conv(
            x, training=training) if self.residual_conv else x
        return self.activation(self.conv(x, training=training) + residual)


class ResidualLayer3D(ResidualLayer3DBase):
    CONV = {
        "standard": tf.keras.layers.Conv3D,
        "bispectral": BSHConv3D,
        "spectral": SSHConv3D,
    }

    def __init__(
        self,
        filters,
        kernel_size,
        use_batch_norm=True,
        conv_type="standard",
        n_conv=1,
        padding='SAME',
        **kwargs,
    ):
        self.conv_constructor = ResidualLayer3D.CONV[conv_type]
        super().__init__(
            filters,
            kernel_size,
            use_batch_norm=use_batch_norm,
            n_conv=n_conv,
            padding=padding,
            **kwargs,
        )

    def _get_conv_block(self):
        kwargs = {k: i for k, i in self.kwargs.items() if k != "activation"}
        block = tf.keras.Sequential()
        for i in range(self.n_conv):
            block.add(
                self.conv_constructor(self.filters,
                                      self.kernel_size,
                                      padding=self.padding,
                                      **kwargs))
            if self.use_batch_norm:
                block.add(tf.keras.layers.BatchNormalization())

            if i != self.n_conv - 1:
                block.add(tf.keras.layers.Activation(self.activation))
        return block

    def _get_residual_conv_block(self):
        block = tf.keras.Sequential()
        block.add(
            tf.keras.layers.Conv3D(
                self.filters,
                1,
                activation="linear",
                strides=self.kwargs.get("strides", 1),
            ))
        if self.padding == "VALID":
            block.add(
                tf.keras.layers.Cropping3D(cropping=(
                    self.n_conv * (self.kernel_size[0] // 2),
                    self.n_conv * (self.kernel_size[1] // 2),
                    self.n_conv * (self.kernel_size[2] // 2),
                )))

        if self.use_batch_norm:
            block.add(tf.keras.layers.BatchNormalization())
        return block


class ResidualGLayer3D(ResidualLayer3DBase):

    def build(self, input_shape):
        self.c_in = input_shape[-2]
        if input_shape[-2] != self.filters:
            self.residual_conv = self._get_residual_conv_block()

    def _get_conv_block(self):
        kwargs = {k: i for k, i in self.kwargs.items() if k != "activation"}
        block = tf.keras.Sequential()
        for i in range(self.n_conv):
            block.add(
                GroupConv(self.filters,
                          self.kernel_size,
                          padding=self.padding,
                          **kwargs))
            if self.use_batch_norm:
                block.add(tf.keras.layers.BatchNormalization(axis=-2))

            if i != self.n_conv - 1:
                block.add(tf.keras.layers.Activation(self.activation))
        return block

    def _get_residual_conv_block(self):
        block = tf.keras.Sequential()
        block.add(
            GSkip(
                self.filters,
                self.kernel_size,
                padding=self.padding,
                kernel_initializer=self.kwargs.get("kernel_initializer",
                                                   "glorot_uniform"),
                n_conv=self.n_conv,
            ))
        block.add(tf.keras.layers.BatchNormalization(axis=-2))
        return block


class GSkip(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_initializer="glorot_uniform",
                 padding="VALID",
                 n_conv=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=0.2)

        self.filters = filters
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (
            kernel_size, kernel_size, kernel_size)
        self.padding = padding.upper()
        self.crop = None
        self.repeat = False
        self.n_conv = n_conv

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, 1, 1, 1, input_shape[-2],
                                        self.filters, 1),
                                 initializer=self.kernel_initializer,
                                 name='w_residual')
        if self.padding == "VALID":
            self.crop = tf.keras.layers.Cropping3D(cropping=(
                self.n_conv * (self.kernel_size[0] // 2),
                self.n_conv * (self.kernel_size[1] // 2),
                self.n_conv * (self.kernel_size[2] // 2),
            ))
        if input_shape[-1] == 1:
            self.repeat = True

    def call(self, x):
        batch_size, depth, height, width, channels, group_dims = x.get_shape(
        ).as_list()

        if self.crop is not None:
            x = tf.reshape(x, (
                batch_size,
                depth,
                height,
                width,
                channels * group_dims,
            ))
            x = self.crop(x)
            _, depth, height, width, _ = tf.shape(x)
            x = tf.reshape(x, (
                batch_size,
                depth,
                height,
                width,
                channels,
                group_dims,
            ))
        x = tf.expand_dims(x, axis=-2)
        x = tf.reduce_sum(x * self.w, axis=-3)

        # if self.repeat:
        #     # when the it is the lifting layer, we need to repeat the output
        #     x = tf.repeat(x, group_dims, axis=1)

        return x
