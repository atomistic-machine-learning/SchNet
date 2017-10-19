import numpy as np
import tensorflow as tf


def shape(x):
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    return np.shape(x)
