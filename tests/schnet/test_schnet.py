import pytest

import numpy as np
import tensorflow as tf


@pytest.fixture
def placeholders():
    shapes = {
        'numbers_list': (None, None),
        'positions_list': (None, None, 3),
        'seg_m': (None,),
        'idx_ik': (None,),
        'seg_i': (None,),
        'idx_j': (None,),
        'idx_jk': (None,),
        'seg_j': (None,),
        'ratio_j': (None,),
        'numbers': (None,),
        'positions': (None, 3),
        'offset': (None, 3),
        'cells': (None, 3, 3)
    }
    dtypes = {
        'numbers_list': tf.int64,
        'positions_list': tf.float32,
        'numbers': tf.int64,
        'positions': tf.float32,
        'offset': tf.float32,
        'seg_m': tf.int64,
        'seg_i': tf.int64,
        'seg_j': tf.int64,
        'idx_ik': tf.int64,
        'idx_jk': tf.int64,
        'ratio_j': tf.float32
    }

    ph = {
        name: tf.placeholder(dt, shape=shapes[name], name=name)
        for name, dt in dtypes.items()
    }
    return ph


@pytest.fixture
def molecules():
    Zlist = [np.array([1, 1]), np.array([6, 6, 6])]
    Rlist = [np.random.rand(2, 3).astype(np.float32),
             np.random.rand(3, 3).astype(np.float32)]
    Z = np.array([1, 1, 6, 6, 6])
    R = np.vstack(Rlist)
    off = np.zeros((8, 3), dtype=np.float32)

    seg_m = np.array([0, 0, 1, 1, 1], dtype=np.int)
    seg_i = np.array([0, 1, 2, 2, 3, 3, 4, 4], dtype=np.int)
    seg_j = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int)
    idx_ik = np.array([0, 1, 2, 2, 3, 3, 4, 4], dtype=np.int)
    idx_jk = np.array([1, 0, 3, 4, 2, 4, 2, 3], dtype=np.int)
    ratio_j = np.array([1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    mols = {
        'numbers_list': Zlist,
        'positions_list': Rlist,
        'numbers': Z,
        'positions': R,
        'offset': off,
        'seg_m': seg_m,
        'seg_i': seg_i,
        'seg_j': seg_j,
        'idx_ik': idx_ik,
        'idx_jk': idx_jk,
        'ratio_j': ratio_j
    }
    return mols
