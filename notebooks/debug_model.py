import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from src.data.tf_data import TFDataCreator
from src.data.data_augmentation import preprocess_ct
from src.models.models import Unet

project_dir = Path(__file__).resolve().parents[1]
file = h5py.File(
    project_dir / "data/processed/Task08_HepaticVessel_training.hdf5", "r")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ds = TFDataCreator(file,
                   patch_size=(64, 64, 64),
                   random_position=True,
                   shuffle=True).get_tf_data().map(lambda x, y: (preprocess_ct(
                       x, clip_value_min=-100, clip_value_max=200), y))

yo = ds.batch(4).as_numpy_iterator()

x, y = next(yo)

model = Unet(output_channels=1,
             last_activation="sigmoid",
             n_features=[12, 24, 48, 96, 192])

y_pred = model(x)
print("The end")
