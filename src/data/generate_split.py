import logging
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import h5py
import SimpleITK as sitk
from sklearn.model_selection import RepeatedStratifiedKFold

project_dir = Path(__file__).resolve().parents[2]
path_data = project_dir / f"data/processed/Task08_HepaticVessel_training.hdf5"

n_rep = 10


def main():
    file = h5py.File(path_data, 'r')
    patient_ids = list(file.keys())
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    id_dicts = []
    n_validation = int(np.round(0.2 * len(patient_ids)))
    for train_index, test_index in cv.split(np.zeros((len(patient_ids), 1)),
                                            np.zeros((len(patient_ids), ))):
        training_ids = [patient_ids[i] for i in train_index]
        testing_ids = [patient_ids[i] for i in test_index]
        print(train_index, test_index)
        id_dicts.append({
            "training": training_ids[n_validation:],
            "validation": training_ids[:n_validation],
            "testing": testing_ids
        })

    with open(
            project_dir /
            "data/processed/Task08_HepaticVessel_training_split.json",
            'w') as f:
        json.dump(id_dicts, f)


if __name__ == '__main__':
    main()
