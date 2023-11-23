from pathlib import Path
import json
from itertools import product

import numpy as np
from tqdm import tqdm
import h5py
import SimpleITK as sitk


def main():
    """
    Execute this Python script to preprocess images from the Medical Decathlon
    dataset (http://medicaldecathlon.com/).
    To enable proper functioning, ensure the unzip folder of the Medical
    Decathlon data is placed in the 'data/raw' directory.
    This script performs resampling on the images using a 
    linear interpolator for images and a nearest neighbor interpolator 
    for label images.
    The processed images are then stored in an HDF5 file, facilitating faster image
    retrieval during training.
    Two subroutines, namely 'process_training' and 'process_test', have been implemented.
    The distinction arises from the fact that in the testing split, labels are not provided,
    leading to a slightly different logic.
    """

    resampling = (1.0, 1.0, 1.0)
    project_dir = Path(__file__).resolve().parents[2]
    path_raw = project_dir / "data/raw"
    process_dict = {
        "training": process_training,
        "test": process_test,
    }

    tasks = [x.name for x in path_raw.iterdir() if x.is_dir()]
    # tasks = ["Task04_Hippocampus"]
    splits = ["training", "test"]

    for split, task in product(splits, tasks):
        print(f"Processing the {split} split of task {task} -- START")
        output_file = project_dir / f"data/processed/{task}/{task}_{split}.hdf5"
        process_dict[split](project_dir, task, output_file, resampling)
        print(f"Processing the {split} split of task {task} -- END")


def process_test(project_dir, task, output_file, resampling):
    split = "test"

    path_data = project_dir / f"data/raw/{task}"
    with open(path_data / "dataset.json") as f:
        meta = json.load(f)

    image_ids = [(n, i.split("/")[-1].split(".")[0])
                 for n, i in enumerate(meta[split])]

    if output_file.exists():
        output_file.unlink()  # delete file if exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hdf5_file = h5py.File(output_file, 'a')
    for image_number, image_id in tqdm(image_ids):
        image_path = path_data / meta[split][image_number]
        image_sitk = sitk.ReadImage(str(image_path))

        if image_sitk.GetSpacing() != resampling:
            image_sitk = resample(image_sitk,
                                  resampling=resampling,
                                  interpolator=sitk.sitkLinear)

        image = np.transpose(sitk.GetArrayFromImage(image_sitk), (2, 1, 0))

        hdf5_file.create_group(f"{image_id}")
        hdf5_file.create_dataset(f"{image_id}/image",
                                 data=image,
                                 dtype="float32")

    hdf5_file.close()


def process_training(project_dir, task, output_file, resampling):
    split = "training"

    path_data = project_dir / f"data/raw/{task}"
    with open(path_data / "dataset.json") as f:
        meta = json.load(f)

    image_ids = [(n, i["image"].split("/")[-1].split(".")[0])
                 for n, i in enumerate(meta[split])]

    if output_file.exists():
        output_file.unlink()  # delete file if exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hdf5_file = h5py.File(output_file, 'a')
    for image_number, image_id in tqdm(image_ids):
        image_path = path_data / meta[split][image_number]["image"]
        label_path = path_data / meta[split][image_number]["label"]
        image_sitk = sitk.ReadImage(str(image_path))
        label_sitk = sitk.ReadImage(str(label_path))

        if image_sitk.GetSpacing() != resampling:
            image_sitk = resample(image_sitk,
                                  resampling=resampling,
                                  interpolator=sitk.sitkLinear)
            label_sitk = resample(label_sitk,
                                  resampling=resampling,
                                  interpolator=sitk.sitkNearestNeighbor)

        image = np.transpose(sitk.GetArrayFromImage(image_sitk), (2, 1, 0))
        label = np.transpose(sitk.GetArrayFromImage(label_sitk), (2, 1, 0))

        hdf5_file.create_group(f"{image_id}")
        hdf5_file.create_dataset(f"{image_id}/image",
                                 data=image,
                                 dtype="float32")
        hdf5_file.create_dataset(f"{image_id}/label",
                                 data=label,
                                 dtype="uint8")

    hdf5_file.close()


def resample(image, resampling=(1.0, 1.0, 1.0), interpolator=None):

    if interpolator is None:
        interpolator = sitk.sitkLinear
    bb = get_bouding_boxes(image)
    size = np.round((bb[3:] - bb[:3]) / np.array(resampling)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(resampling)
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def get_bouding_boxes(image):
    """
    Get the bounding boxes of the CT and PT images.
    This works since all images have the same direction
    """

    origin = np.array(image.GetOrigin())

    position_max = origin + np.array(image.GetSize()) * np.array(
        image.GetSpacing())
    return np.concatenate([origin, position_max], axis=0)


if __name__ == '__main__':
    main()
