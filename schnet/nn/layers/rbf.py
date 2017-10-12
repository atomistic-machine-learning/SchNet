import numpy as np
import tensorflow as tf

from ..module import Module


class RBFExpansion(Module):
    def __init__(self, dim, cutoff, gap, lower_cutoff=0., name=None):
        self.cutoff = cutoff
        self.gap = gap
        self.dim = dim
        xrange = cutoff - lower_cutoff
        self.centers = np.linspace(lower_cutoff, self.cutoff,
                                   int(np.ceil(xrange / self.gap)))
        self.n_centers = len(self.centers)
        self.fan_out = self.dim * self.n_centers
        super(RBFExpansion, self).__init__(name)

    def _forward(self, d):
        cshape = tf.shape(d)
        CS = d.get_shape()
        centers = self.centers.reshape((1, 1, 1, 1, -1)).astype(np.float32)
        d = tf.expand_dims(d, -1) - tf.constant(centers)
        rbf = tf.exp(-(d ** 2) / self.gap)
        rbf = tf.reshape(rbf, (
            cshape[0], cshape[1], cshape[2],
            self.dim * centers.shape[-1]))
        rbf.set_shape([CS[0], CS[1], CS[2], self.dim * self.n_centers])
        return rbf
