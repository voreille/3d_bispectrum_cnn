import tensorflow as tf
import numpy as np

from src.models.cubenet.layers import GroupConv
from src.models.models import ResidualGLayer3D
from src.models.models import GUnet
from src.models.utils import config_gpu

config_gpu(gpu_id="0", memory_limit=16)
# layer = SHConv3D.get("radial")(5, max_degree=5, padding="same")
# layer = ResidualGLayer3D(7, 3, group="S4, padding="valid")
layer = GUnet(n_features=(4, 8, 16, 32, 64), group="S4")
# layer.build((32, 32, 32, 3, 1))

image_size = 64

image = tf.random.uniform((2, image_size, image_size, image_size, 2))
output = layer(image)

print(output.shape)
