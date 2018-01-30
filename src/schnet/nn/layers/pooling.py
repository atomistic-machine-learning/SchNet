import tensorflow as tf

from .module import Module


class PoolSegments(Module):
    def __init__(self, mode='sum', name=None):
        if mode == 'sum':
            self._reduce = tf.segment_sum
        elif mode == 'mean':
            self._reduce = tf.segment_mean
        super(PoolSegments, self).__init__(name)

    def _forward(self, x, segs):
        num_idx = segs[-1] + 1
        g = tf.get_default_graph()
        num_idx = tf.Print(num_idx, [num_idx, tf.shape(x)])
        with g.gradient_override_map({"Tile": "TileDense"}):
            y = self._reduce(x, segs)
        return y
