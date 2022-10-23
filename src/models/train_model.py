from pathlib import Path
import os
import datetime
import logging
import pprint

import click
import yaml
import h5py
from dotenv import find_dotenv, load_dotenv
import tensorflow as tf
import pandas as pd

from src.models.models import get_compiled_model
from src.data.tf_data import TFDataCreator
from src.models.utils import config_gpu
from src.data.utils import get_split
from src.models.callbacks import EarlyStopping
from src.models.losses import dice_coefficient_hard

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).resolve().parents[2]

config_path = project_dir / "configs/config.yaml"

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
def main(config, gpu_id, memory_limit, split_id, output_path, log_path):
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
    ids_test = get_split(split_id, os.environ["SPLITPATH"])["testing"]

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
    ).batch(config["training"]["batch_size"])
    if config["training"]["steps_per_epoch"]:
        ds_train = ds_train.shuffle(buffer_size=128).repeat().take(
            config["training"]["steps_per_epoch"])

    ds_val = tf_data_creator.get_tf_data(
        ids_val,
        data_augmentation=False,
    ).batch(config["training"]["batch_size"])

    callbacks = get_callbacks(config)
    model_name = get_model_name(config, split_id)
    if not DEBUG:
        log_dir = project_dir / (log_path + "/fit/" + model_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    model = get_compiled_model(config["model"], run_eagerly=run_eagerly)
    model.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
    )

    ds_train = tf_data_creator.get_tf_data_with_id(ids_train).batch(
        config["training"]["batch_size"])
    ds_val = tf_data_creator.get_tf_data_with_id(ids_val).batch(
        config["training"]["batch_size"])
    ds_test = tf_data_creator.get_tf_data_with_id(ids_test).batch(
        config["training"]["batch_size"])

    logger.info(f"Evaluating on training, validation and testing"
                f" set for split {split_id}")
    results_train = evaluate_model(model, ds_train)
    results_val = evaluate_model(model, ds_val)
    results_test = evaluate_model(model, ds_test)
    logger.info(f"Evaluating on training, validation and testing"
                f" set for split {split_id} - DONE")
    file.close()

    if not DEBUG:
        save_stuff(config, model, model_name, split_id, output_path,
                   results_train, results_val, results_test)


def get_model_name(config, split_id):
    nf = "".join([str(n) + "_" for n in config["model"]["n_features"]])
    return (f"{config['model']['model_name']}__"
            f"ks_{config['model']['kernel_size']}__"
            f"nf_{nf}__split_{split_id}"
            f"__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")


def save_stuff(config, model, output_name, split_id, output_path,
               results_train, results_val, results_test):
    output_folder = Path(output_path) / output_name
    output_folder.mkdir(parents=True, exist_ok=True)
    dir_to_save_weights = output_folder / "weights" / f"split_{split_id}" / "final"
    model.save_weights(dir_to_save_weights)

    with open(output_folder / "config.yaml", "w") as f:
        yaml.dump(config, f)

    with open(output_folder / "architecture.txt", "w") as f:
        model.summary(print_fn=lambda s: print(s, file=f))

    results_train.to_csv(output_folder /
                         f"results_train__split_{split_id}.csv")
    results_val.to_csv(output_folder / f"results_val__split_{split_id}.csv")
    results_test.to_csv(output_folder / f"results_test__split_{split_id}.csv")


def evaluate_model(model, ds_test):
    results = pd.DataFrame()
    i = 0
    for x, y, image_ids in ds_test:
        y_pred = model(x)
        dices_1 = dice_coefficient_hard(y[..., 1], y_pred[..., 1]).numpy()
        dices_2 = dice_coefficient_hard(y[..., 2], y_pred[..., 2]).numpy()
        for b in range(y.shape[0]):
            results = pd.concat([
                results,
                pd.DataFrame(
                    {
                        "image_id": image_ids[b].numpy().decode("utf-8"),
                        "dice_1": dices_1[b],
                        "dice_2": dices_2[b],
                    },
                    index=[i],
                )
            ])
            i += 1
    return results


def get_callbacks(config):
    callbacks = list()
    config_lr_decay = config["training"]["lr_scheduler"]

    def scheduler(epoch, lr):
        return lr * (
            1 - epoch / config["training"]["epochs"])**config_lr_decay["power"]

    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

    # callbacks.append(
    #     EarlyStopping(
    #         monitor='val_loss',
    #         min_delta=0,
    #         patience=10,
    #         verbose=0,
    #         mode='min',
    #         restore_best_weights=True,
    #         minimal_num_of_epochs=50,
    #     ))
    return callbacks


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
