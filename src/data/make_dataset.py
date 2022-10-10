from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import h5py
import SimpleITK as sitk

split = "training"
task = "Task04_Hippocampus"

project_dir = Path(__file__).resolve().parents[2]
path_data = project_dir / f"data/raw/{task}"
output_file = project_dir / f"data/processed/{task}/{task}_{split}.hdf5"

with open(path_data / "dataset.json") as f:
    meta = json.load(f)

resampling = (1.0, 1.0, 1.0)


def main():
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
            print(f"Resampling image {image_id}")
            image_sitk, label_sitk = resample(image_sitk, label_sitk)

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


def resample(image, label):

    bb = get_bouding_boxes(image)
    size = np.round((bb[3:] - bb[:3]) / np.array(resampling)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(resampling)
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    resampler.SetInterpolator(sitk.sitkLinear)
    image = resampler.Execute(image)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    label = resampler.Execute(label)
    return image, label


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
