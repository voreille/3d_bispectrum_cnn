import random

import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter, rotate


def preprocess_ct(image, clip_value_min=-45, clip_value_max=105):
    image = tf.clip_by_value(image, clip_value_min, clip_value_max)
    return (image - clip_value_min) / (clip_value_max - clip_value_min)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, s_min=0.1, s_max=2.):
        self.prob = p
        self.s1_min = s_min
        self.s1_max = s_max

    def __call__(self, image):
        image_shape = image.shape
        image = tf.py_function(self.call, [image], tf.float32)
        image.set_shape(image_shape)
        return image

    def call(self, img):
        if hasattr(img, "numpy"):
            img = img.numpy()

        do_it = random.random() <= self.prob
        if not do_it:
            return img
        img[..., 0] = gaussian_filter(
            img[..., 0],
            sigma=random.uniform(self.s1_min, self.s1_max),
        )

        return img


class RightAngleRotation(object):
    """
    Apply the same transformation to a liste of 3D images.
    """

    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, image, segmentation):
        image_shape = image.shape
        segmentation_shape = segmentation.shape
        image, segmentation = tf.py_function(self.call, [image, segmentation],
                                             [tf.float32, tf.float32])
        image.set_shape(image_shape)
        segmentation.set_shape(segmentation_shape)
        return image, segmentation

    def call(self, image, segmentation):
        do_it = random.random() <= self.prob
        if not do_it:
            return image, segmentation
        permutation = random.sample([0, 1, 2], 3)
        permutation.append(3)
        flips = np.random.randint(2, size=(3, )) == 1

        image = self.apply(image, permutation, flips)
        segmentation = self.apply(segmentation, permutation, flips)
        return image, segmentation

    def apply(self, img, permutation, flips):
        if hasattr(img, "numpy"):
            img = img.numpy()

        img = np.transpose(img, permutation)
        for i in range(3):
            if flips[i]:
                img = np.flip(img, axis=i)
        return img
