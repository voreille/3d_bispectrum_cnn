from pathlib import Path

import tensorflow as tf
import yaml

from src.models.layers import BSHConv3D
from src.models.losses import dice_loss, dice_coefficient_hard


def load_model(model_dir, split_id=0, run_eagerly=False):
    """
    Loads a model from a path.
    """

    model_dir = Path(model_dir).resolve()

    with open(model_dir / "config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = get_compiled_model(config["model"], run_eagerly=run_eagerly)
    path_weights = model_dir / "weights" / f"split_{split_id}" / "final"
    model.load_weights(str(path_weights))
    return model


def get_compiled_model(params, run_eagerly=False):
    model_name = params["model_name"]
    model_params = {
        k: i
        for k, i in params.items()
        if k in ["output_channels", "n_features", "last_activation"]
    }
    if model_name == "Unet":
        model = Unet(**model_params)
    elif model_name == "BSHConv3D":
        model = BSHConv3D(**model_params["model"])
    else:
        raise ValueError(f"Model {model_name} not implemented")

    return compile_model(model, params["compile"], run_eagerly=run_eagerly)


def dice_vessel(y_true, y_pred):
    return dice_coefficient_hard(y_true[..., 1], y_pred[..., 1])


def dice_tumor(y_true, y_pred):
    return dice_coefficient_hard(y_true[..., 2], y_pred[..., 2])


def compile_model(model, params, run_eagerly=False):
    if params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(
            params["learning_rate"]), )
    else:
        raise ValueError(f"Optimizer {params['optimizer']} not implemented")

    if params["loss"] == "dsc":
        loss = lambda y_true, y_pred: dice_loss(y_true[..., 0], y_pred[
            ..., 0]) + dice_loss(y_true[..., 1], y_pred[..., 1]) + dice_loss(
                y_true[..., 2], y_pred[..., 2])
    else:
        raise ValueError(f"Loss {params['loss']} not implemented")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[dice_vessel, dice_tumor],
        run_eagerly=run_eagerly,
    )

    return model


class UnetBase(tf.keras.Model):

    def __init__(
        self,
        output_channels=1,
        last_activation="sigmoid",
        n_features=None,
    ):
        super().__init__()
        if n_features is None:
            n_features = [12, 24, 48, 96, 192]
        self.output_channels = output_channels
        self.last_activation = last_activation
        self.stem = self.get_first_block(n_features[0])
        self.down_stack = [
            self.get_residual_block(n_features[1]),
            self.get_residual_block(n_features[2]),
            self.get_residual_block(n_features[3]),
            self.get_residual_block(n_features[4]),
        ]

        self.up_stack = [
            self.get_conv_block(n_features[4]),
            self.get_conv_block(n_features[3]),
            self.get_conv_block(n_features[2]),
            self.get_conv_block(n_features[1]),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(output_channels,
                                   1,
                                   activation=last_activation,
                                   padding='SAME'),
        ])
        self.max_pool_stack = [
            tf.keras.layers.MaxPool3D(),
            tf.keras.layers.MaxPool3D(),
            tf.keras.layers.MaxPool3D(),
            tf.keras.layers.MaxPool3D(),
        ]
        self.upsampling_stack = [
            self.get_upsampling_block(n_features[3]),
            self.get_upsampling_block(n_features[2]),
            self.get_upsampling_block(n_features[1]),
            self.get_upsampling_block(n_features[0]),
        ]

    def get_first_block(self, filters):
        raise NotImplementedError("Must be implemented in subclass")

    def get_residual_block(self, filters):
        raise NotImplementedError("Must be implemented in subclass")

    def get_conv_block(self, filters):
        raise NotImplementedError("Must be implemented in subclass")

    def get_upsampling_block(self, filters):
        raise NotImplementedError("Must be implemented in subclass")

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        x = self.stem(x)
        for block, max_pool in zip(self.down_stack, self.max_pool_stack):
            skips.append(x)
            x = block(x, training=training)
            x = max_pool(x, training=training)

        skips = reversed(skips)

        for block, skip, upsample in zip(self.up_stack, skips,
                                         self.upsampling_stack):
            x = block(
                tf.keras.layers.concatenate([
                    upsample(x, training=training),
                    skip,
                ]),
                training=training,
            )

        return self.last(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channels": self.output_channels,
            "last_activation": self.last_activation,
            "n_features": self.n_features,
        })
        return config


class Unet(UnetBase):

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters, 7, padding='SAME', activation="relu"),
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def get_residual_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def get_conv_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters,
                                   3,
                                   padding='SAME',
                                   activation="relu"),
            tf.keras.layers.Conv3D(filters,
                                   3,
                                   padding='SAME',
                                   activation="relu"),
        ])

    def get_upsampling_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.UpSampling3D(),
            # tf.keras.layers.Conv3D(filters,
            #                        3,
            #                        padding='SAME',
            #                        activation="relu"),
        ])


class BLRIUnet(UnetBase):

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualBLRILayer3D(filters, 7, padding='SAME', activation="relu"),
            ResidualBLRILayer3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def get_residual_block(self, filters):
        return tf.keras.Sequential([
            ResidualBLRILayer3D(filters, 3, padding='SAME', activation="relu"),
            ResidualBLRILayer3D(filters, 3, padding='SAME', activation="relu"),
            ResidualBLRILayer3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def get_conv_block(self, filters):
        return tf.keras.Sequential([
            BSHConv3D(filters, 3, padding='SAME', activation="relu"),
            BSHConv3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def get_upsampling_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.UpSampling3D(),
            # BSHConv3D(filters, 3, padding='SAME', activation="relu"),
        ])


class ResidualLayerBase(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.conv = None
        self.residual_conv = None
        self.activation = None

    def call(self, x, training=None):
        residual = self.residual_conv(
            x, training=training) if self.residual_conv else x
        return self.activation(self.conv(x, training=training) + residual)


class ResidualLayer3D(ResidualLayerBase):

    def __init__(self, *args, activation='relu', padding="SAME", **kwargs):
        super().__init__(kwargs.get("name"))
        self.filters = args[0]
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv3D(*args,
                                   **kwargs,
                                   padding=padding,
                                   activation="linear"),
            tf.keras.layers.BatchNormalization(),
        ])
        self.activation = tf.keras.layers.Activation(activation)
        self.strides = kwargs.get("strides", 1)

    def build(self, input_shape):
        self.c_in = input_shape[-1]
        if input_shape[-1] != self.filters:
            self.residual_conv = tf.keras.Sequential([
                tf.keras.layers.Conv3D(self.filters,
                                       1,
                                       activation=self.activation,
                                       strides=self.strides),
                tf.keras.layers.BatchNormalization()
            ])


class ResidualBLRILayer3D(ResidualLayerBase):

    def __init__(self, *args, activation='relu', padding="SAME", **kwargs):
        super().__init__(kwargs.get("name"))
        self.filters = args[0]
        self.conv = tf.keras.Sequential([
            BSHConv3D(*args, **kwargs, padding=padding, activation="linear"),
            tf.keras.layers.BatchNormalization(),
        ])
        self.activation = tf.keras.layers.Activation(activation)
        self.strides = kwargs.get("strides", 1)

    def build(self, input_shape):
        self.c_in = input_shape[-1]
        if input_shape[-1] != self.filters:
            self.residual_conv = tf.keras.Sequential([
                tf.keras.layers.Conv3D(self.filters,
                                       1,
                                       activation=self.activation,
                                       strides=self.strides),
                tf.keras.layers.BatchNormalization()
            ])