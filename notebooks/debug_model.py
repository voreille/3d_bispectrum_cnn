import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from src.data.tf_data import TFDataCreator
from src.data.data_augmentation import preprocess_ct
from src.models.models import Unet, SLRIUnet
from src.models.utils import config_gpu

project_dir = Path(__file__).resolve().parents[1]
file = h5py.File(
    project_dir /
    "data/processed/Task04_Hippocampus/Task04_Hippocampus_training.hdf5", "r")

config_gpu("0", memory_limit=16)
ds = TFDataCreator(file, patch_size=(64, 64, 64),
                   shuffle=True).get_tf_data().map(lambda x, y: (preprocess_ct(
                       x, clip_value_min=-100, clip_value_max=200), y))
ds_creator = TFDataCreator.get("Task04")(file,
                                         params_augmentation={
                                             "rotation": False,
                                             "random_center": False,
                                         })
ds = ds_creator.get_tf_data()

yo = ds.batch(4).as_numpy_iterator()

x, y = next(yo)

model = SLRIUnet(output_channels=1,
                 last_activation="sigmoid",
                 n_features=[12, 24, 48, 96, 192])

y_pred = model(x)
print("The end")
