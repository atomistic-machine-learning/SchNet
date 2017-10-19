import numpy as np
import tensorflow as tf

from .module import Module
from ..initializers import glorot_uniform
from ..utils import shape


class Dense(Module):
    '''
      Fully-connected layer
      y = nonlinearity(Wx+b)

    '''

    def __init__(self, fan_in, fan_out,
                 use_bias=True, activation=None,
                 w_init=glorot_uniform, b_init=tf.zeros_initializer(),
                 trainable=True, name=None):
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._use_bias = use_bias
        self._w_init = w_init
        self._b_init = b_init
        self.activation = activation
        self._trainable = trainable
        super(Dense, self).__init__(name)

    def _initialize(self):
        if type(self._w_init) is np.ndarray:
            shape = None
            self._w_init = self._w_init.astype(np.float32)
        else:
            shape = (self._fan_in, self._fan_out)
        self.W = tf.get_variable('W', shape=shape,
                                 initializer=self._w_init,
                                 trainable=self._trainable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.W)
        tf.summary.histogram('W', self.W)

        if self._use_bias:
            self.b = tf.get_variable('b', shape=(self._fan_out,),
                                     initializer=self._b_init,
                                     trainable=self._trainable)
            tf.add_to_collection(tf.GraphKeys.BIASES, self.b)
            tf.summary.histogram('b', self.b)

    def _forward(self, x):
        x_shape = shape(x)
        ndims = len(x_shape)

        # reshape for broadcasting
        assert x_shape[-1] == self._fan_in
        xr = tf.reshape(x, (-1, self._fan_in))
        y = tf.matmul(xr, self.W)

        if self._use_bias:
            y += self.b

        if self.activation:
            y = self.activation(y)

        new_shape = tf.concat([tf.shape(x)[:ndims - 1], [self._fan_out]],
                              axis=0)
        y = tf.reshape(y, new_shape)
        new_dims = x_shape[:-1] + [self._fan_out]
        y.set_shape(new_dims)
        tf.summary.histogram('activations', y)
        return y
