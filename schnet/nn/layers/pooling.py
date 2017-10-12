import tensorflow as tf

from ..module import Module


class PoolSegments(Module):
    def __init__(self, mode='sum', name=None):
        if mode == 'sum':
            self._reduce = tf.segment_sum
        elif mode == 'mean':
            self._reduce = tf.segment_mean
        super(PoolSegments, self).__init__(name)

    def _forward(self, x, ids):
        y = self._reduce(x, ids)
        return y
