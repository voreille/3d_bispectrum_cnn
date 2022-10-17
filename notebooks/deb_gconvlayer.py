import tensorflow as tf
import numpy as np

from src.models.cubenet.layers import GroupConv
from src.models.models import ResidualGLayer3D
from src.models.utils import config_gpu

config_gpu(gpu_id="0", memory_limit=10)
# layer = SHConv3D.get("radial")(5, max_degree=5, padding="same")
layer = ResidualGLayer3D(7, 3, group="S4", padding="valid")
layer.build((32, 32, 32, 3, 1))

image_size = 32
dirac = np.zeros((2, image_size, image_size, image_size, 3, 1))
dirac[0, image_size // 2, image_size // 2, image_size // 2, :, 0] = 1
dirac[1, image_size // 2, image_size // 2, image_size // 2, :, 0] = 0.5

# image = tf.random.uniform((2, 32, 32, 32, 2))
output = layer(dirac)

print(output.shape)
