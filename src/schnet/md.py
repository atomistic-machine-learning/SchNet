import os
import numpy as np
import tensorflow as tf

from .models.schnet import SchNet


def get_atom_indices(n_atoms, batch_size):
    n_distances = n_atoms ** 2 - n_atoms
    seg_m = np.repeat(range(batch_size), n_atoms).astype(np.int32)
    seg_i = np.repeat(np.arange(n_atoms * batch_size), n_atoms - 1).astype(np.int32)
    idx_ik = seg_i
    idx_j = []
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_atoms):
                if j != i:
                    idx_j.append(j + b * n_atoms)

    idx_j = np.hstack(idx_j).ravel().astype(np.int32)
    offset = np.zeros((n_distances * batch_size, 3), dtype=np.float32)
    ratio_j = np.ones((n_distances * batch_size,), dtype=np.float32)
    seg_j = np.arange(n_distances * batch_size, dtype=np.int32)

    seg_m, idx_ik, seg_i, idx_j, seg_j, offset, ratio_j = \
        tf.constant(seg_m), tf.constant(idx_ik), tf.constant(seg_i), tf.constant(idx_j), \
        tf.constant(seg_j), tf.constant(offset), tf.constant(ratio_j)
    idx_jk = idx_j
    return seg_m, idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j


class SchNetMD:
    def __init__(self, energy_model_path, force_model_path=None, batch_size=1,
                 nuclear_charges=6. * np.ones((20,), dtype=np.int64)):

        self.n_atoms = len(nuclear_charges)
        seg_m, idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = get_atom_indices(self.n_atoms, batch_size)

        self.energy_model = self.load_model(energy_model_path)
        self.force_model = self.load_model(force_model_path) \
            if force_model_path is not None else None

        self.positions = tf.placeholder(tf.float32, shape=(batch_size * self.n_atoms, 3))
        self.charges = tf.tile(tf.constant(nuclear_charges.ravel(), dtype=tf.int64),
                               (batch_size,))

        g = tf.get_default_graph()
        with g.gradient_override_map({"Tile": "TileDense"}):
            self.energy = self.energy_model(self.charges, self.positions,
                                            offset, idx_ik, idx_jk, idx_j,
                                            seg_m, seg_i, seg_j, ratio_j)
            if self.force_model is None:
                self.forces = -tf.reshape(tf.convert_to_tensor(
                    tf.gradients(tf.reduce_sum(self.energy), self.positions)[0]),
                    (batch_size, self.n_atoms, 3))
            else:
                energy = self.force_model(self.charges, self.positions,
                                          offset, idx_ik, idx_jk, idx_j,
                                          seg_m, seg_i, seg_j, ratio_j)
                self.forces = -tf.reshape(tf.convert_to_tensor(tf.gradients(tf.reduce_sum(energy),
                                                                            self.positions)[0]),
                                          (batch_size, self.n_atoms, 3))
        self.error = tf.reduce_max(tf.sqrt(tf.reduce_sum(self.forces ** 2, 2)))

        ckpt = tf.train.latest_checkpoint(os.path.join(energy_model_path, 'validation'))
        self.session = tf.Session()
        self.energy_model.restore(self.session, ckpt)
        if self.force_model is not None:
            ckpt = tf.train.latest_checkpoint(os.path.join(force_model_path, 'validation'))
            self.force_model.restore(self.session, ckpt)

    def load_model(self, model_path):
        args = np.load(os.path.join(model_path, 'args.npy')).item()

        model = SchNet(args.interactions, args.basis, args.filters, args.cutoff,
                       intensive=args.intensive, filter_pool_mode=args.filter_pool_mode)
        return model

    def get_energy_and_forces(self, positions):
        positions = positions.reshape((-1, 3)).astype(np.float32)
        feed_dict = {
            self.positions: positions
        }
        E, F = self.session.run([self.energy, self.forces], feed_dict=feed_dict)
        return E, F

    def relax(self, positions, eps=0.01, rate=1e-4):
        err = 100.
        positions = positions.reshape((-1, 3)).astype(np.float32)
        print('Start relaxation')
        count = 0
        while err > eps:
            feed_dict = {
                self.positions: positions
            }
            F, err = self.session.run([self.forces, self.error], feed_dict=feed_dict)
            positions += rate * F[0]
            count += 1
            if count % 100 == 0:
                print('Iteration ', str(count), 'Max Force:', err)
        print('Maximal force length: ' + str(err))
        return positions
