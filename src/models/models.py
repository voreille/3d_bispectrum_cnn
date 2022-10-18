from pathlib import Path
from matplotlib.pyplot import axis

import tensorflow as tf
import yaml

from src.models.layers_faster import BSHConv3D, SSHConv3D
from src.models.residual_layers import ResidualGLayer3D, ResidualLayer3D
from src.models.losses import dice_loss, dice_coefficient_hard
from src.models.cubenet.layers import GroupConv, MaxPoolG, UpsamplingG


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
        for k, i in params.items() if k in
        ["output_channels", "n_features", "last_activation", "kernel_size"]
    }
    if model_name == "Unet":
        model = Unet(conv_type="standard", **model_params)
    elif model_name == "BLRIUnet":
        model = Unet(conv_type="bispectral", **model_params)
    elif model_name == "SLRIUnet":
        model = Unet(conv_type="spectral", **model_params)
    elif model_name == "DebugNet":
        model = DebugNet(**model_params)
    elif model_name == "S4Unet":
        model = GUnet(group="S4", **model_params)
    elif model_name == "T4Unet":
        model = GUnet(group="T4", **model_params)
    else:
        raise ValueError(f"Model {model_name} not implemented")

    return compile_model(model,
                         params["compile"],
                         output_channels=params["output_channels"],
                         run_eagerly=run_eagerly)


def dice_0(y_true, y_pred):
    return dice_coefficient_hard(y_true[..., 0], y_pred[..., 0])


def dice_1(y_true, y_pred):
    return dice_coefficient_hard(y_true[..., 1], y_pred[..., 1])


def dice_2(y_true, y_pred):
    return dice_coefficient_hard(y_true[..., 2], y_pred[..., 2])


def crossentropy(y_true, y_pred):
    l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(l, axis=(1, 2, 3))


def compile_model(model, params, output_channels=3, run_eagerly=False):
    if params["optimizer"] == "adam":
        # lr = tf.keras.optimizers.schedules.ExponentialDecay(
        #     **params["lr_scheduler"], )
        optimizer = tf.keras.optimizers.Adam(params["learning_rate"])
    else:
        raise ValueError(f"Optimizer {params['optimizer']} not implemented")

    if params["loss"] == "dsc":
        loss = lambda y_true, y_pred: tf.reduce_mean(
            dice_loss(
                y_true[..., 1],
                y_pred[..., 1],
            ) + dice_loss(
                y_true[..., 2],
                y_pred[..., 2],
            ) + crossentropy(
                y_true,
                y_pred,
            ))
        # loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        raise ValueError(f"Loss {params['loss']} not implemented")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[dice_0, dice_1, dice_2],
        run_eagerly=run_eagerly,
    )

    return model


class DebugNet(tf.keras.Model):

    def __init__(
        self,
        output_channels=2,
        last_activation="sigmoid",
        n_features=None,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.last_activation = last_activation
        self.stem = self.get_first_block(12)
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(
                output_channels,
                1,
                activation=last_activation,
                dtype=tf.float32,
                padding='SAME',
            ),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            SSHConv3D(filters,
                      7,
                      max_degree=2,
                      padding='SAME',
                      activation="relu"),
        ])

    def call(self, x, training=False):
        x = self.stem(x)
        x = self.last(x)
        return x


class Unet(tf.keras.Model):
    CONV = {
        "standard": tf.keras.layers.Conv3D,
        "bispectral": BSHConv3D,
        "spectral": SSHConv3D,
    }

    def __init__(
        self,
        output_channels=1,
        last_activation="sigmoid",
        n_features=None,
        kernel_size=3,
        conv_type="standard",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        if n_features is None:
            n_features = [12, 24, 48, 96, 192]
        self.output_channels = output_channels
        self.last_activation = last_activation
        self.conv_type = conv_type
        self.conv_constructor = self.CONV[conv_type]
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
            tf.keras.layers.Conv3D(
                output_channels,
                1,
                activation=last_activation,
                dtype=tf.float32,
                padding='SAME',
            ),
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
            "conv_type": self.conv_type,
        })
        return config

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters,
                            7,
                            padding='SAME',
                            activation="relu",
                            conv_type=self.conv_type),
            ResidualLayer3D(filters,
                            self.kernel_size,
                            padding='SAME',
                            activation="relu",
                            conv_type=self.conv_type),
        ])

    def get_residual_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters,
                            self.kernel_size,
                            padding='SAME',
                            activation="relu",
                            conv_type=self.conv_type),
            ResidualLayer3D(filters,
                            self.kernel_size,
                            padding='SAME',
                            activation="relu",
                            conv_type=self.conv_type),
            ResidualLayer3D(filters,
                            self.kernel_size,
                            padding='SAME',
                            activation="relu",
                            conv_type=self.conv_type),
        ])

    def get_conv_block(self, filters):
        return tf.keras.Sequential([
            self.conv_constructor(filters,
                                  self.kernel_size,
                                  padding='SAME',
                                  activation="relu"),
            self.conv_constructor(filters,
                                  self.kernel_size,
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


class GUnet(tf.keras.Model):

    def __init__(
        self,
        output_channels=1,
        last_activation="sigmoid",
        n_features=None,
        kernel_size=3,
        group="S4",
        use_batch_norm=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.group = group
        self.use_batch_norm = use_batch_norm

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
            tf.keras.layers.Conv3D(
                output_channels,
                1,
                activation=last_activation,
                dtype=tf.float32,
                padding='SAME',
            ),
        ])
        self.max_pool_stack = [
            MaxPoolG(),
            MaxPoolG(),
            MaxPoolG(),
            MaxPoolG(),
        ]
        self.upsampling_stack = [
            self.get_upsampling_block(n_features[3]),
            self.get_upsampling_block(n_features[2]),
            self.get_upsampling_block(n_features[1]),
            self.get_upsampling_block(n_features[0]),
        ]

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        x = tf.expand_dims(x, axis=-1)
        x = self.stem(x)
        for block, max_pool in zip(self.down_stack, self.max_pool_stack):
            skips.append(x)
            x = block(x, training=training)
            x = max_pool(x, training=training)

        skips = list(reversed(skips))

        for block, skip, upsample in zip(self.up_stack, skips,
                                         self.upsampling_stack):
            x = block(
                tf.keras.layers.concatenate(
                    [
                        upsample(x, training=training),
                        skip,
                    ],
                    axis=-2,
                ),
                training=training,
            )

        return self.last(tf.reduce_mean(x, axis=-1), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channels": self.output_channels,
            "last_activation": self.last_activation,
            "n_features": self.n_features,
            "conv_type": self.conv_type,
        })
        return config

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualGLayer3D(filters,
                             7,
                             padding='SAME',
                             activation="relu",
                             group=self.group),
            ResidualGLayer3D(filters,
                             self.kernel_size,
                             padding='SAME',
                             activation="relu",
                             group=self.group),
        ])

    def get_residual_block(self, filters):
        return tf.keras.Sequential([
            ResidualGLayer3D(filters,
                             self.kernel_size,
                             padding='SAME',
                             activation="relu",
                             group=self.group),
            ResidualGLayer3D(filters,
                             self.kernel_size,
                             padding='SAME',
                             activation="relu",
                             group=self.group),
            ResidualGLayer3D(filters,
                             self.kernel_size,
                             padding='SAME',
                             activation="relu",
                             group=self.group),
        ])

    def get_conv_block(self, filters, n_conv=2):
        block = tf.keras.Sequential()
        for _ in range(n_conv):
            block.add(
                GroupConv(
                    filters,
                    self.kernel_size,
                    padding='SAME',
                    activation="linear",
                    group=self.group,
                ))
            if self.use_batch_norm:
                block.add(tf.keras.layers.BatchNormalization(axis=-2))
            block.add(tf.keras.layers.Activation("relu"))
        return block

    def get_upsampling_block(self, filters):
        return UpsamplingG()
