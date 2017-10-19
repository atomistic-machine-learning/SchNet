import tensorflow as tf

from .module import Module
from .dense import Dense
from .pooling import PoolSegments


class CFConv(Module):
    """
    Continuous-filter convolution layer
    """

    def __init__(self, fan_in, fan_out, n_filters, pool_mode='sum',
                 activation=None, name=None):
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._n_filters = n_filters
        self.activation = activation
        self.pool_mode = pool_mode
        super(CFConv, self).__init__(name=name)

    def _initialize(self):
        self.in2fac = Dense(self._fan_in, self._n_filters, use_bias=False,
                            name='in2fac')
        self.fac2out = Dense(self._n_filters, self._fan_out, use_bias=True,
                             activation=self.activation,
                             name='fac2out')
        self.pool = PoolSegments(mode=self.pool_mode)

    def _forward(self, x, w, seg_i, idx_j):
        '''
        :param x (num_atoms, num_feats): input
        :param w (num_interactions, num_filters): filters
        :param seg_i (num_interactions,): segments of atom i
        :param idx_j: (num_interactions,): indices of atom j
        :return: convolution x * w
        '''
        # to filter-space
        f = self.in2fac(x)

        # filter-wise convolution
        f = tf.gather(f, idx_j)
        wf = w * f
        conv = self.pool(wf, seg_i)

        # to output-space
        y = self.fac2out(conv)
        return y
