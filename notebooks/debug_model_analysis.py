import os

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.models.models import load_model
from src.data.tf_data import TFDataCreator
from src.data.utils import get_split
from src.models.losses import dice_loss, dice_coefficient_hard
from src.models.utils import config_gpu

load_dotenv(find_dotenv())

config_gpu(0, 16)

task = "Task04_Hippocampus"

model_path = "models/S4Unet__ks_3__nf_4_8_16_32_64___split_0__20221018-172719"

model = load_model(model_path)

split_id = 0
ids_train = get_split(split_id, os.environ["SPLITPATH"])["training"]
ids_val = get_split(split_id, os.environ["SPLITPATH"])["validation"]
ids_test = get_split(split_id, os.environ["SPLITPATH"])["testing"]

file = h5py.File(f"./data/processed/{task}/{task}_training.hdf5", "r")
data_creator = TFDataCreator.get(task.split("_")[0])(
    file,
    #    patch_size=(128, 128, 128),
    shuffle=True,
    params_augmentation={
        "rotation": False,
        "random_center": False,
    })
ds = data_creator.get_tf_data_with_id(ids_test).batch(4)

x, y, image_id = next(ds.as_numpy_iterator())


y_pred = model(x)
print("The model predicts: ", y_pred.shape)
