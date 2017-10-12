import numpy as np
import tensorflow as tf

from ..module import Module
from .dense import Dense
from .pooling import PoolSegments

from ..initializers import glorot_uniform
from ..utils import shape


class EuclideanDistances(Module):
    def __init__(self, name=None):
        super(EuclideanDistances, self).__init__(name)

    def _forward(self, r, cells, idx_i, idx_j):
        p1 = tf.expand_dims(r, 1)
        p2 = tf.expand_dims(r, 2)
        Rij = p1 - p2
        Dij2 = tf.reduce_sum(Rij ** 2, -1)
        dshape = tf.shape(Dij2)
        Dij2 += tf.eye(num_rows=dshape[1], batch_shape=[dshape[0]])
        Dij = tf.sqrt(tf.nn.relu(Dij2))
        Dij = tf.expand_dims(Dij, -1)
        return Dij
