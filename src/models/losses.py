import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


class DiceLoss(tf.keras.losses.Loss):

    def __init__(self,
                 loss_type="jaccard",
                 smooth=1e-6,
                 spatial_axis=(1, 2, 3),
                 name="dice_loss"):
        super().__init__(name=name)
        self.loss_type = loss_type
        self.smooth = smooth
        self.spatial_axis = spatial_axis

    def call(self, y_true, y_pred):
        return dice_loss(
            y_true,
            y_pred,
            loss_type=self.loss_type,
            smooth=self.smooth,
            spatial_axis=self.spatial_axis,
        )


def dice_coefficient_hard(y_true,
                          y_pred,
                          loss_type='sorensen',
                          smooth=1e-6,
                          spatial_axis=(1, 2, 3)):
    return dice_coefficient(y_true,
                            tf.cast(y_pred > 0.5, tf.float32),
                            loss_type=loss_type,
                            smooth=smooth,
                            spatial_axis=spatial_axis)


def dice_loss(
        y_true,
        y_pred,
        loss_type='jaccard',
        smooth=1e-6,
        spatial_axis=(1, 2, 3),
        reduction=None,
):
    l = 1 - dice_coefficient(
        y_true,
        y_pred,
        loss_type=loss_type,
        smooth=smooth,
        spatial_axis=spatial_axis,
    )

    if reduction == "mean":
        return tf.reduce_mean(l)

    return l


def dice_coefficient(y_true,
                     y_pred,
                     loss_type='jaccard',
                     smooth=1e-6,
                     spatial_axis=(1, 2, 3)):
    intersection = tf.reduce_sum(y_true * y_pred, axis=spatial_axis)
    if loss_type == 'jaccard':
        union = tf.reduce_sum(
            tf.square(y_pred),
            axis=spatial_axis,
        ) + tf.reduce_sum(tf.square(y_true), axis=spatial_axis)

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred, axis=spatial_axis) + tf.reduce_sum(
            y_true, axis=spatial_axis)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)
    return (2. * intersection + smooth) / (union + smooth)


@tf.function
def sigmoid_focal_crossentropy(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    from_logits: bool = False,
) -> tf.Tensor:
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.

    Copied from https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/losses/focal_loss.py#L84-L142
    and modified to avoid reduction
    """
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = tf.keras.losses.binary_crossentropy(y_true,
                                             y_pred,
                                             from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return, modified by Val
    return alpha_factor * modulating_factor * ce
