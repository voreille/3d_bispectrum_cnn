from operator import index
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import h5py
import SimpleITK as sitk
import pandas as pd

split = "training"

project_dir = Path(__file__).resolve().parents[2]
path_data = project_dir / "data/raw/Task04_Hippocampus"
# output_file = project_dir / f"data/processed/Task08_HepaticVessel_{split}.csv"
output_file = project_dir / f"data/processed/Task04_Hippocampus_{split}.csv"

with open(path_data / "dataset.json") as f:
    meta = json.load(f)


def get_size_bb(label_sitk):
    label_array = sitk.GetArrayFromImage(label_sitk)
    bb = np.argwhere(label_array)
    bb_min = np.min(bb, axis=0)
    bb_max = np.max(bb, axis=0)
    return bb_max - bb_min + 1


def main():
    image_ids = [(n, i["image"].split("/")[-1].split(".")[0])
                 for n, i in enumerate(meta[split])]

    df = pd.DataFrame()
    for image_number, image_id in tqdm(image_ids):
        image_path = path_data / meta[split][image_number]["image"]
        label_path = path_data / meta[split][image_number]["label"]
        image_sitk = sitk.ReadImage(str(image_path))
        label_sitk = sitk.ReadImage(str(label_path))
        bb_size = get_size_bb(label_sitk)
        spacing = image_sitk.GetSpacing()
        direction = image_sitk.GetDirection()
        size = image_sitk.GetSize()
        spacing_dict = {f"spacing_{i}": spacing[i] for i in range(3)}
        bb_size_dict = {f"size_bb_{i}": bb_size[i] for i in range(3)}
        direction_dict = {f"direction_{i}": direction[i] for i in range(9)}
        size_dict = {f"size_{i}": size[i] for i in range(3)}
        output_dict = dict(
            image_id=image_id,
            **spacing_dict,
            **size_dict,
            **bb_size_dict,
        )
        #    **direction_dict)
        df = pd.concat([df, pd.DataFrame(output_dict, index=[image_number])])
    print(df.describe())
    df.to_csv(output_file)


if __name__ == '__main__':
    main()
