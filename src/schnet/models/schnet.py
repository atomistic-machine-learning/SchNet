import numpy as np
import schnet.nn.layers as L
import tensorflow as tf
from schnet.nn.activation import shifted_softplus


class SchNetFilter(L.Module):
    def __init__(self, n_in, n_filters,
                 pool_mode='sum', name=None):
        self.n_in = n_in
        self.pool_mode = pool_mode
        self.n_filters = n_filters

        super(SchNetFilter, self).__init__(name=name)

    def _initialize(self):
        self.dense1 = L.Dense(self.n_in, self.n_filters,
                              activation=shifted_softplus)
        self.dense2 = L.Dense(self.n_filters, self.n_filters,
                              activation=shifted_softplus)
        self.pooling = L.PoolSegments(self.pool_mode)

    def _forward(self, dijk, seg_j, ratio_j=1.):
        h = self.dense1(dijk)
        w_ijk = self.dense2(h)
        w_ij = self.pooling(w_ijk, seg_j)
        if self.pool_mode == 'mean':
            w_ij *= ratio_j

        return w_ij


class SchNetInteractionBlock(L.Module):
    def __init__(self, n_in, n_basis, n_filters, pool_mode='sum',
                 name=None):
        self.n_in = n_in
        self.n_basis = n_basis
        self.n_filters = n_filters
        self.pool_mode = pool_mode
        super(SchNetInteractionBlock, self).__init__(name=name)

    def _initialize(self):
        self.filternet = SchNetFilter(self.n_in, self.n_filters,
                                      pool_mode=self.pool_mode)
        self.cfconv = L.CFConv(
            self.n_basis, self.n_basis, self.n_filters,
            activation=shifted_softplus
        )
        self.dense = L.Dense(self.n_basis, self.n_basis)

    def _forward(self, x, dijk, idx_j, seg_i, seg_j, ratio_j=None):
        w = self.filternet(dijk, seg_j, ratio_j)
        h = self.cfconv(x, w, seg_i, idx_j)
        v = self.dense(h)
        y = x + v
        return y

    def _calc_filter(self, dijk, seg_j, ratio_j):
        w = self.filternet(dijk, seg_j, ratio_j)
        return w


class SchNet(L.Module):
    def __init__(self, n_interactions, n_basis, n_filters, cutoff,
                 mean_per_atom=np.zeros((1,), dtype=np.float32),
                 std_per_atom=np.ones((1,), dtype=np.float32),
                 gap=0.1, atomref=None, intensive=False,
                 filter_pool_mode='sum',
                 shared_interactions=False,
                 n_embeddings=100, name=None):
        self.n_interactions = n_interactions
        self.n_basis = n_basis
        self.n_filters = n_filters
        self.n_embeddings = n_embeddings
        self.cutoff = cutoff
        self.shared_interactions = shared_interactions
        self.intensive = intensive
        self.filter_pool_mode = filter_pool_mode
        self.atomref = atomref
        self.gap = gap

        self.mean_per_atom = mean_per_atom
        self.std_per_atom = std_per_atom
        super(SchNet, self).__init__(name=name)

    def _initialize(self):
        self.atom_embedding = L.Embedding(
            self.n_embeddings, self.n_basis, name='atom_embedding'
        )

        self.dist = L.EuclideanDistances()
        self.rbf = L.RBFExpansion(0., self.cutoff, self.gap)

        if self.shared_interactions:
            self.interaction_blocks = \
                [
                    SchNetInteractionBlock(
                        self.rbf.fan_out, self.n_basis, self.n_filters,
                        pool_mode=self.filter_pool_mode,
                        name='interaction')
                ] * self.n_interactions
        else:
            self.interaction_blocks = [
                SchNetInteractionBlock(
                    self.rbf.fan_out, self.n_basis, self.n_filters,
                    name='interaction_' + str(i))
                for i in range(self.n_interactions)
            ]

        self.dense1 = L.Dense(self.n_basis, self.n_basis // 2,
                              activation=shifted_softplus)
        self.dense2 = L.Dense(self.n_basis // 2, 1,
                              w_init=tf.zeros_initializer())
        if self.intensive:
            self.atom_pool = L.PoolSegments('mean')
        else:
            self.atom_pool = L.PoolSegments('sum')

        self.mean_per_atom = tf.get_variable('mean_per_atom',
                                             initializer=tf.convert_to_tensor(
                                                 np.array(self.mean_per_atom,
                                                          dtype=np.float32)),
                                             trainable=False)
        self.std_per_atom = tf.get_variable('std_per_atom',
                                            initializer=tf.convert_to_tensor(
                                                np.array(self.std_per_atom,
                                                         dtype=np.float32)),
                                            trainable=False)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self.mean_per_atom)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self.std_per_atom)

        if self.atomref is not None:
            self.e0 = L.Embedding(self.n_embeddings, 1, trainable=False,
                                  embedding_init=self.atomref, name='atomref')
        else:
            self.e0 = L.Embedding(self.n_embeddings, 1, trainable=False,
                                  embedding_init=tf.zeros_initializer(),
                                  name='atomref')

    def _forward(self, z, r, offsets, idx_ik, idx_jk,
                 idx_j, seg_m, seg_i, seg_j, ratio_j):
        # embed atom species
        x = self.atom_embedding(z)

        # interaction features
        dijk = self.dist(r, offsets, idx_ik, idx_jk)
        dijk = self.rbf(dijk)

        # interaction blocks
        for iblock in self.interaction_blocks:
            x = iblock(x, dijk, idx_j, seg_i, seg_j, ratio_j)
            # x = print_shape(x)

        # output network
        h = self.dense1(x)
        y_i = self.dense2(h)
        # y_i = print_shape(y_i)

        # scale energy contributions
        y_i = y_i * self.std_per_atom + self.mean_per_atom

        if self.e0 is not None:
            y_i += self.e0(z)

        y = self.atom_pool(y_i, seg_m)
        return y

    def get_filters(self, r, offsets, idx_ik, idx_jk, seg_j, ratio_j):
        dijk = self.dist(r, offsets, idx_ik, idx_jk)
        dijk = self.rbf(dijk)

        filters = []
        for iblock in self.interaction_blocks:
            filters.append(iblock._calc_filter(dijk, seg_j, ratio_j))
        return filters


def print_shape(t, name=None):
    if name is None:
        name = t.name
    return tf.Print(t, [tf.shape(t), t], summarize=20, message=name)
