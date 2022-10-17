import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

from src.models.layers_faster import LinearUpsampling3D
from src.models.models import ResidualLayer3D
from src.models.utils import config_gpu

config_gpu(0, 4)

image = np.random.uniform(size=(1, 16, 16, 16, 2))

# layer = LinearUpsampling3D(size=(2, 4, 6))
layer = ResidualLayer3D(8, 3, padding="valid", activation="relu")

image_utf = layer(image)

print(f"output shape: {image_utf.shape}")
