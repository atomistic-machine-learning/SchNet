import numpy as np
import tensorflow as tf

from .module import Module
from .dense import Dense


class Embedding(Module):
    def __init__(self, n_embeddings, dim,
                 embedding_init=None,
                 trainable=True,
                 name=None):
        self._n_embeddings = n_embeddings
        self._dim = dim
        self._embedding_init = embedding_init
        self._trainable = trainable
        super(Embedding, self).__init__(name)

    def _initialize(self):
        if self._embedding_init is None:
            r = tf.sqrt(1. / tf.sqrt(float(self._dim)))
            self._embedding_init = tf.random_normal_initializer(stddev=r)

        self.embeddings = Dense(self._n_embeddings, self._dim, use_bias=False,
                                w_init=self._embedding_init,
                                trainable=self._trainable,
                                name='embeddings')

    def _forward(self, indices):
        I = np.eye(self._n_embeddings).astype(np.float32)
        ind = tf.nn.embedding_lookup(I, indices)
        y = self.embeddings(ind)
        return y
