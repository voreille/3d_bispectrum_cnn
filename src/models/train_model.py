from pathlib import Path
import os
import datetime

import click
import yaml
import h5py
from dotenv import find_dotenv, load_dotenv
import tensorflow as tf

from src.models.models import get_compiled_model
from src.data.tf_data import TFDataCreator
from src.models.utils import config_gpu
from src.data.utils import get_split
from src.models.callbacks import EarlyStopping

project_dir = Path(__file__).resolve().parents[2]

config_path = project_dir / "configs/config.yaml"

DEBUG = False

if DEBUG:
    run_eagerly = True
else:
    run_eagerly = False


@click.command()
@click.option("--config",
              type=click.Path(exists=True),
              default=config_path,
              help="config file")
@click.option("--gpu-id", type=click.STRING, default="1", help="gpu id")
@click.option("--split-id", type=click.INT, default=0, help="split id")
@click.option(
    "--output-path",
    type=click.Path(exists=False),
    default="models",
    help="split id",
)
def main(config, gpu_id, split_id, output_path):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_gpu(gpu_id, config["gpu"]["memory_limit"])
    file = h5py.File(os.environ["DATAPATH"], 'r')
    ids_train = get_split(split_id, os.environ["SPLITPATH"])["training"]
    ids_val = get_split(split_id, os.environ["SPLITPATH"])["validation"]
    tf_data_creator = TFDataCreator(
        file,
        image_ids=ids_train,
        patch_size=config["data"]["patch_size"],
        clip_value_min=config["preprocessing"]["clip_value_min"],
        clip_value_max=config["preprocessing"]["clip_value_max"],
        num_parallel_calls=tf.data.AUTOTUNE,
        params_augmentation=config["data"]["augmentation"],
    )
    ds_train = tf_data_creator.get_tf_data(
        ids_train,
        data_augmentation=True,
    ).batch(config["training"]["batch_size"])

    ds_val = tf_data_creator.get_tf_data(
        ids_val,
        data_augmentation=False,
    ).batch(4)

    callbacks = [
        EarlyStopping(
            minimal_num_of_epochs=10,
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='max',
            restore_best_weights=True,
        )
    ]
    model = get_compiled_model(config["model"], run_eagerly=run_eagerly)
    model.fit(x=ds_train,
              validation_data=ds_val,
              epochs=config["training"]["epochs"],
              callbacks=callbacks)

    file.close()

    if not DEBUG:
        save_stuff(config, model, split_id, output_path)


def save_stuff(config, model, split_id, output_path):
    output_name = (f"{config['model']['model_name']}__split_{split_id}"
                   f"__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    output_folder = Path(output_path) / output_name
    output_folder.mkdir(parents=True, exist_ok=True)
    dir_to_save_weights = output_folder / "weights" / f"split_{split_id}" / "final"
    model.save_weights(dir_to_save_weights)

    with open(output_folder / "config.yaml", "w") as f:
        yaml.dump(config, f)

    with open(output_folder / "architecture.txt", "w") as f:
        model.summary(print_fn=lambda s: print(s, file=f))


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
