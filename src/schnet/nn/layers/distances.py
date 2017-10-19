import tensorflow as tf

from .module import Module


class EuclideanDistances(Module):
    def __init__(self, name=None):
        super(EuclideanDistances, self).__init__(name)

    def _forward(self, r, offsets, idx_ik, idx_jk):
        ri = tf.gather(r, idx_ik)
        rj = tf.gather(r, idx_jk) + offsets
        rij = ri - rj

        dij2 = tf.reduce_sum(rij ** 2, -1, keep_dims=True)
        dij = tf.sqrt(tf.nn.relu(dij2))
        return dij
