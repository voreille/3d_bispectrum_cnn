import random

import tensorflow as tf
import numpy as np
from numpy.random import randint

from src.data.data_augmentation import RightAngleRotation


class TFDataCreator:

    def __init__(
        self,
        file,
        patch_size=(64, 64, 64),
        num_parallel_calls=None,
        image_ids=None,
        shuffle=False,
        clip_value_min=-100,
        clip_value_max=200,
        params_augmentation=None,
    ):

        self.file = file
        self.patch_size = patch_size
        self.num_parallel_calls = num_parallel_calls
        if image_ids is None:
            self.image_ids = list(file.keys())
        else:
            self.image_ids = image_ids
        self.shuffle = shuffle
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        if params_augmentation is None:
            self.params_augmentation = {
                "rotation": True,
                "random_center": True
            }
        else:
            self.params_augmentation = params_augmentation

    def _get_origin_from_center(self, center, *, image_shape):
        dx = self.patch_size[0] // 2
        dy = self.patch_size[1] // 2
        dz = self.patch_size[2] // 2
        origin = [center[0] - dx, center[1] - dy, center[2] - dz]
        for i in range(3):
            if origin[i] < 0:
                origin[i] = 0
            if origin[i] + self.patch_size[i] > image_shape[i]:
                origin[i] = image_shape[i] - self.patch_size[i]
        return origin

    def _get_random_center(self, label):
        positions = np.where(label != 0)
        idx = randint(0, len(positions[0]))
        return positions[0][idx], positions[1][idx], positions[2][idx]

    def _get_center(self, label):
        positions = np.where(label != 0)
        x_min = np.min(positions[0])
        y_min = np.min(positions[1])
        z_min = np.min(positions[2])

        x_max = np.max(positions[0])
        y_max = np.max(positions[1])
        z_max = np.max(positions[2])

        return (x_min + x_max) // 2, (y_min + y_max) // 2, (z_min + z_max) // 2

    def _preprocess(self, image):
        image = tf.clip_by_value(image, self.clip_value_min,
                                 self.clip_value_max)
        return 2 * (image - self.clip_value_min) / (self.clip_value_max -
                                                    self.clip_value_min) - 1

    def _get_patch(self, image, label, center):
        x, y, z = self._get_origin_from_center(center, image_shape=image.shape)

        image_cropped = image[x:x + self.patch_size[0],
                              y:y + self.patch_size[1],
                              z:z + self.patch_size[2]]
        label_cropped = label[x:x + self.patch_size[0],
                              y:y + self.patch_size[1],
                              z:z + self.patch_size[2]]
        label_cropped = np.stack([
            label_cropped == 0,
            label_cropped == 1,
            label_cropped == 2,
        ],
                                 axis=-1)
        return image_cropped, label_cropped

    def _parse_image(self, image_id):
        if hasattr(image_id, "numpy"):
            image_id = image_id.numpy().decode("utf-8")
        image = self.file[image_id]["image"][()]
        label = self.file[image_id]["label"][()]
        center = self._get_center(label)
        image_cropped, label_cropped = self._get_patch(image, label, center)

        return self._preprocess(image_cropped[..., np.newaxis]), label_cropped

    def _parse_image_random_center(self, image_id):
        if hasattr(image_id, "numpy"):
            image_id = image_id.numpy().decode("utf-8")
        image = self.file[image_id]["image"][()]
        label = self.file[image_id]["label"][()]
        center = self._get_random_center(label)
        image_cropped, label_cropped = self._get_patch(image, label, center)

        return self._preprocess(image_cropped[..., np.newaxis]), label_cropped

    def tf_parse_image(self, image_id):
        image, label = tf.py_function(self._parse_image, [image_id],
                                      [tf.float32, tf.float32])
        image.set_shape(tuple(self.patch_size) + (1, ))
        label.set_shape(tuple(self.patch_size) + (3, ))
        return image, label

    def tf_parse_image_random_center(self, image_id):
        image, label = tf.py_function(self._parse_image_random_center,
                                      [image_id], [tf.float32, tf.float32])
        image.set_shape(tuple(self.patch_size) + (1, ))
        label.set_shape(tuple(self.patch_size) + (3, ))
        return image, label

    def get_tf_data(self, image_ids=None, data_augmentation=False):
        if image_ids:
            ds = tf.data.Dataset.from_tensor_slices(image_ids)
        else:
            ds = tf.data.Dataset.from_tensor_slices(self.image_ids)

        if self.shuffle:
            ds.shuffle(150)

        if data_augmentation:
            ds = ds.map(self.tf_parse_image_random_center,
                        num_parallel_calls=self.num_parallel_calls)
            if self.params_augmentation.get("rotation", False):
                data_augmenter = RightAngleRotation(p=1.0)
                ds = ds.map(data_augmenter,
                            num_parallel_calls=self.num_parallel_calls)
        else:
            ds = ds.map(self.tf_parse_image,
                        num_parallel_calls=self.num_parallel_calls)

        return ds
