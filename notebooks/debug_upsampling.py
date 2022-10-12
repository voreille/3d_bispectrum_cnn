import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

from src.models.layers_faster import LinearUpsampling3D
from src.models.utils import config_gpu

config_gpu(0, 4)

image = np.random.uniform(size=(1, 16, 16, 16, 2))

layer = LinearUpsampling3D(size=(2, 4, 6))

image_utf = layer(image)

image_utf.shape
