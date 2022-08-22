import os

import tensorflow as tf
import numpy as np

from src.models.layers import SSHConv3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

layer = SSHConv3D(1,
                  17,
                  max_degree=5,
                  padding="same",
                  initializer=tf.keras.initializers.Constant(value=1.0),
                  project=False)
layer.build((32, 32, 32, 2))

image_size = 32
dirac = np.zeros((1, image_size, image_size, image_size, 1))
dirac[0, image_size // 2, image_size // 2, image_size // 2, 0] = 1

# image = tf.random.uniform((2, 32, 32, 32, 2))
output = layer(dirac)

print(output.shape)
