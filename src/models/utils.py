import os

import tensorflow as tf


def config_gpu(gpu_id, memory_limit=None):
    """
    Configures the GPU to use the most available memory.
    memory_limit: float, memory limit in GB
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate of memory on the first GPU
        try:
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0], [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=memory_limit * 1024)
                    ])
            else:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
