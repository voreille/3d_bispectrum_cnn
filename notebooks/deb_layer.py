import os

import tensorflow as tf
import numpy as np

from src.models.layers_faster import SSHConv3D, SHConv3D, BSHConv3D
from src.models.utils import config_gpu

# config_gpu(gpu_id="0", memory_limit=10)
# layer = SHConv3D.get("radial")(5, max_degree=5, padding="same")
layer = BSHConv3D(2,
                  5,
                  max_degree=3,
                  padding="valid",
                  kernel_initializer=tf.keras.initializers.Constant(value=1.0),
                  project=False)
layer.build((32, 32, 32, 3))

image_size = 32
dirac = np.zeros((1, image_size, image_size, image_size, 3))
dirac[0, image_size // 2, image_size // 2, image_size // 2, :] = 1

# image = tf.random.uniform((2, 32, 32, 32, 2))
output = layer(dirac)

print(output.shape)
