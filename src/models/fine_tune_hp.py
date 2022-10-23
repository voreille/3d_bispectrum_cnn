"""
TODO:
TODO: lr scheduler and early stopping?
"""

from pathlib import Path
import os
import logging
import pprint

import click
import yaml
import h5py
from dotenv import find_dotenv, load_dotenv
import tensorflow as tf
import keras_tuner as kt

from src.models.models import get_compiled_model
from src.data.tf_data import TFDataCreator
from src.models.utils import config_gpu
from src.data.utils import get_split

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).resolve().parents[2]

config_path = project_dir / "configs/hptuning_config.yaml"

DEBUG = False
run_eagerly = False
pp = pprint.PrettyPrinter(depth=4)


@click.command()
@click.option("--config",
              type=click.Path(exists=True),
              default=config_path,
              help="config file")
@click.option("--gpu-id", type=click.STRING, default="0", help="gpu id")
@click.option("--memory-limit",
              type=click.FLOAT,
              default=None,
              help="GPU memory limit in GB")
@click.option("--split-id", type=click.INT, default=0, help="split id")
@click.option(
    "--output-path",
    type=click.Path(exists=False),
    default="models",
    help="Relative path to save the model",
)
@click.option(
    "--log-path",
    type=click.Path(exists=False),
    default="logs",
    help="Relative path to save the logs",
)
@click.option("--epoch-multiplier", type=click.INT, default=25, help="gpu id")
def main(config, gpu_id, memory_limit, split_id, output_path, log_path,
         epoch_multiplier):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Config:")
    pp.pprint(config)

    config_gpu(gpu_id, memory_limit=memory_limit)

    if config["model"]["mixed_precision"]:
        logger.info("Using mixed precision")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    file = h5py.File(os.environ["DATAPATH"], 'r')
    ids_train = get_split(split_id, os.environ["SPLITPATH"])["training"]
    ids_val = get_split(split_id, os.environ["SPLITPATH"])["validation"]
    tf_data_creator = TFDataCreator.get("Task04")(
        file,
        image_ids=ids_train,
        # patch_size=config["data"]["patch_size"],
        num_parallel_calls=tf.data.AUTOTUNE,
        params_augmentation=config["data"]["augmentation"],
    )
    ds_train = tf_data_creator.get_tf_data(
        ids_train,
        data_augmentation=config["data"]["data_augmentation"],
    ).repeat(epoch_multiplier).batch(config["training"]["batch_size"])

    ds_val = tf_data_creator.get_tf_data(
        ids_val,
        data_augmentation=False,
    ).batch(config["training"]["batch_size"])

    # model = build_model(kt.HyperParameters())
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=15,
        factor=3,
        hyperband_iterations=3,
        directory="results_tuning",
        project_name="task04_standard_unet",
    )

    tuner.search(x=ds_train, epochs=15, validation_data=ds_val)

    file.close()


def build_model(hp):
    n_features_base = hp.Choice("n_features_base", [1, 4, 8, 12, 16])
    config_model = {
        'model_name': 'Unet',
        'linear_upsampling': hp.Boolean("linear_upsampling"),
        'mixed_precision': True,
        'output_channels': 3,
        'kernel_size': hp.Choice("kernel_size", [3, 5, 7]),
        'last_activation': 'softmax',
        'n_features': [2**i * n_features_base for i in range(5)],
        'compile': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss': 'dsc'
        }
    }
    model = get_compiled_model(config_model, run_eagerly=False)
    return model


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
