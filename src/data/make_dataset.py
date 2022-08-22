import logging
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import h5py
import SimpleITK as sitk

split = "training"

project_dir = Path(__file__).resolve().parents[2]
path_data = project_dir / "data/raw/Task08_HepaticVessel"
output_file = project_dir / f"data/processed/Task08_HepaticVessel_{split}.hdf5"

with open(path_data / "dataset.json") as f:
    meta = json.load(f)


def main():
    image_ids = [(n, i["image"].split("/")[-1].split(".")[0])
                 for n, i in enumerate(meta[split])]

    if output_file.exists():
        output_file.unlink()  # delete file if exists
    hdf5_file = h5py.File(output_file, 'a')
    for image_number, image_id in tqdm(image_ids):
        image_path = path_data / meta[split][image_number]["image"]
        label_path = path_data / meta[split][image_number]["label"]
        image_sitk = sitk.ReadImage(str(image_path))
        label_sitk = sitk.ReadImage(str(label_path))

        print(f"Processing image {image_id}")
        print(f"image direction: {image_sitk.GetDirection()}")
        print(f"image spacing: {image_sitk.GetSpacing()}")

        image = np.transpose(sitk.GetArrayFromImage(image_sitk), (2, 1, 0))
        label = np.transpose(sitk.GetArrayFromImage(label_sitk), (2, 1, 0))

        hdf5_file.create_group(f"{image_id}")
        hdf5_file.create_dataset(f"{image_id}/image",
                                 data=image,
                                 dtype="float32")
        hdf5_file.create_dataset(f"{image_id}/label", data=label, dtype="uint16")

    hdf5_file.close()


def parse_image():
    pass


if __name__ == '__main__':
    main()
