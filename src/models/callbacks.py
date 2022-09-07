import tensorflow as tf


class EarlyStopping(tf.keras.callbacks.EarlyStopping):

    def __init__(self, *args, minimal_num_of_epochs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimal_num_of_epochs = minimal_num_of_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.minimal_num_of_epochs:
            super().on_epoch_end(epoch, logs=logs)
