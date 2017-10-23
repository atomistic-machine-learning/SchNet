import os

import numpy as np
import pytest
import schnet
from ase.db import connect
import tensorflow as tf

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../data',
)


@pytest.fixture
def mol_path():
    return os.path.join(DATA_DIR, 'qm9.db')


@pytest.fixture
def atomref():
    path = os.path.join(DATA_DIR, 'atomref.npz')
    atomref = np.load(path)['atom_ref'][:, 1:2]
    return atomref


def test_reader_full_single(mol_path):
    reader = schnet.ASEReader(mol_path, ['energy_U0'], [], [])

    with connect(mol_path) as conn:
        # check length
        assert conn.count() == len(reader)

        # check indexing
        for i in range(20):
            row = conn.get(i + 1)
            assert row['energy_U0'] == reader[i]['energy_U0'], \
                'i=: ' + str(i)
            assert row['energy_U0'] == reader.get_property('energy_U0', i), \
                'i=: ' + str(i)
            assert row['natoms'] == reader.get_number_of_atoms(i), \
                'i=' + str(i)
            assert np.allclose(row['numbers'], reader.get_atomic_numbers(i)), \
                'i=' + str(i)


def test_reader_full_batch(mol_path):
    reader = schnet.ASEReader(mol_path, ['energy_U0'], [], [])

    with connect(mol_path) as conn:
        # check length
        assert conn.count() == len(reader)

        # check batch indexing
        idx = np.random.randint(low=0, high=19, size=(5,))

        data = reader[idx]
        E = reader.get_property('energy_U0', idx)
        Z = reader.get_atomic_numbers(idx)
        N = reader.get_number_of_atoms(idx)
        assert E.shape == (len(idx),)
        assert len(Z) == len(idx)
        assert N.shape == (len(idx),)

        for n, z in zip(N, Z):
            assert n == len(z)

        for k, i in enumerate(idx):
            row = conn.get(int(i) + 1)
            assert row['energy_U0'] == data['energy_U0'][k], 'i=: ' + str(i)
            assert row['energy_U0'] == E[k], 'i=: ' + str(i)
            assert row['natoms'] == N[k], 'i=' + str(i)
            assert np.allclose(row['numbers'], Z[k]), 'i=' + str(i)

        idx = np.arange(2, dtype=np.int)
        data = reader[idx]
        N = reader.get_number_of_atoms(idx)
        seg_m = np.repeat(idx, N)
        idx_ik = np.hstack((np.repeat(np.arange(N[0]), (N[0] - 1,)),
                            np.repeat(np.arange(N[0], N[0] + N[1]),
                                      (N[1] - 1,))))
        idx_jk = [np.setdiff1d(np.arange(N[0]), [i]) for i in np.arange(N[0])]
        idx_jk += [np.setdiff1d(np.arange(N[0], N[0] + N[1]), [i]) for i in
                   np.arange(N[0], N[0] + N[1])]
        idx_jk = np.hstack(idx_jk)
        seg_j = np.arange(np.sum(N ** 2 - N), dtype=np.int)
        ratio_j = np.hstack((np.repeat(1. / (N[0] - 1), N[0] ** 2 - N[0]),
                             np.repeat(1. / (N[1] - 1), N[1] ** 2 - N[1])))

        assert np.all(data['offset'] == 0.)
        assert np.all(data['seg_m'] == seg_m)
        assert np.all(data['seg_i'] == idx_ik)
        assert np.all(data['seg_j'] == seg_j)
        assert np.all(data['idx_ik'] == idx_ik)
        assert np.all(data['idx_jk'] == idx_jk), data['idx_jk']
        assert np.allclose(data['ratio_j'], ratio_j)


def test_data_provider(mol_path):
    reader = schnet.ASEReader(mol_path, ['energy_U0'], [], [])
    sidx = np.random.permutation(range(len(reader)))
    train_idx = sidx[:len(reader)//2]
    val_idx = sidx[len(reader) // 2:]

    assert len(np.intersect1d(train_idx, val_idx)) == 0

    train_provider = schnet.DataProvider(reader, 1, train_idx)
    val_provider = schnet.DataProvider(reader, 1, val_idx, shuffle=False)

    E_train = []
    E_val = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        train_provider.create_threads(sess, coord)
        val_provider.create_threads(sess, coord)

        train_batch = train_provider.get_batch()
        val_batch = val_provider.get_batch()

        for i in range(100):
            E_train.append(sess.run(train_batch)['energy_U0'])
            E_val.append(sess.run(val_batch)['energy_U0'])

    print(E_train)
    print(E_val)
    print(np.intersect1d(E_train, E_val))
    assert len(np.intersect1d(E_train, E_val)) == 0



