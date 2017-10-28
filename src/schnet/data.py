import logging
import threading
from random import shuffle

import numpy as np
import tensorflow as tf
from ase.db import connect

from schnet.atoms import collect_neighbors, IsolatedAtomException


def generate_neighbor_dataset(asedb, nbhdb, cutoff):
    '''
    Generates an ASE DB with neighborhood information from an ASE DB without it.

    :param str asedb: path to original data
    :param str nbhdb: destination
    :param float cutoff: neighborhood cutoff radius
    '''
    skipped = 0
    with connect(asedb, use_lock_file=False) as srccon:
        with connect(nbhdb, use_lock_file=False) as dstcon:
            for row in srccon.select():
                at = row.toatoms()

                try:
                    res = collect_neighbors(at, cutoff)
                except IsolatedAtomException as e:
                    logging.warning('Skipping example  ' + str(skipped) + ' ' +
                                    str(row.id) +
                                    ' due to isolated atom with r_cut=' +
                                    str(cutoff))
                    skipped += 1
                    continue
                idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = res
                try:
                    data = row.data
                except:
                    data = {}
                data['_idx_ik'] = idx_ik
                data['_idx_jk'] = idx_jk
                data['_idx_j'] = idx_j
                data['_seg_i'] = seg_i
                data['_seg_j'] = seg_j
                data['_offset'] = offset
                data['_ratio_j'] = ratio_j
                dstcon.write(at, key_value_pairs=row.key_value_pairs,
                             data=data)


class ASEReader:
    '''
      A reader for a dataset in ASE DB format containing neighborhood
      information and respecting periodic boundary conditions.

      Batches of data can be accessed via indexing.

      Important: Since key_value_pairs are always scalars, their corresponding
                 shape is always be (None, 1), where a batch dimension is added
                 automatically. Data properties are stacked. Their shapes need
                 to be specified and a batch dimension must be added manually in
                 the ASE DB, if needed.

      Args:
          asedb (str): path to data with neighborhood information
          properties (dict): dict from prop names to shape tuples
                              (None for variable dim)
          preload (bool): Preload data to memory, if True (default: False)
          subset (numpy.ndarray): Restricts data to subset specified by indices.
                                  Later indexing is within this subset.
    '''

    def __init__(self, asedb, kvp_properties, data_properties=[],
                 data_shapes=[]):
        self.asedb = asedb
        self.kvp = {k: (None, 1) for k in kvp_properties}
        self.data_props = dict(zip(data_properties, data_shapes))
        self.properties = list(self.kvp.keys()) + list(self.data_props.keys())
        self.data = None

    @property
    def shapes(self):
        shapes = {
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
        shapes.update(self.data_props)
        shapes.update(self.kvp)
        return shapes

    @property
    def dtypes(self):
        data = self[1]
        types = {prop: d.dtype for prop, d in data.items()}
        return types

    @property
    def names(self):
        data = self[1]
        return list(data.keys())

    def __len__(self):
        with connect(self.asedb) as conn:
            return conn.count()

    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]

        with connect(self.asedb) as conn:
            data = {
                'seg_m': [],
                'idx_ik': [],
                'seg_i': [],
                'idx_j': [],
                'idx_jk': [],
                'seg_j': [],
                'numbers': [],
                'positions': [],
                'offset': [],
                'ratio_j': [],
                'cells': []
            }

            for prop in self.properties:
                data[prop] = []

            c_atoms = 0
            c_site_segs = 0
            for k, i in enumerate(idx):
                row = conn.get(int(i) + 1)
                at = row.toatoms()

                if len(row.data['_seg_j']) > 0:
                    upd_site_segs = row.data['_seg_j'][-1] + 1
                else:
                    upd_site_segs = 0

                data['seg_m'].append(np.array([k] * at.get_number_of_atoms()))
                data['idx_ik'].append(row.data['_idx_ik'] + c_atoms)
                data['seg_i'].append(row.data['_seg_i'] + c_atoms)
                data['idx_j'].append(row.data['_idx_j'] + c_atoms)
                data['idx_jk'].append(row.data['_idx_jk'] + c_atoms)
                data['seg_j'].append(row.data['_seg_j'] + c_site_segs)
                data['offset'].append(row.data['_offset'].astype(np.float32))
                data['ratio_j'].append(row.data['_ratio_j'].astype(np.float32))
                data['numbers'].append(at.get_atomic_numbers())
                data['positions'].append(at.get_positions().astype(np.float32))
                data['cells'].append(at.cell[np.newaxis].astype(np.float32))
                c_atoms += at.get_number_of_atoms()
                c_site_segs += upd_site_segs

                for prop in self.kvp.keys():
                    data[prop].append(np.array([[row[prop]]], dtype=np.float32))

                for prop in self.data_props.keys():
                    data[prop].append(row.data[prop], dtype=np.float32)

        data = {p: np.concatenate(b, axis=0) for p, b in data.items()}
        return data

    def get_property(self, pname, idx):
        if type(idx) is int:
            idx = np.array([idx])

        idx = idx + 1

        with connect(self.asedb) as conn:
            property = [conn.get(int(i))[pname] for i in idx]
        property = np.array(property)
        return property

    def get_atomic_numbers(self, idx):
        if type(idx) is int:
            idx = np.array([idx])

        idx = idx + 1

        with connect(self.asedb) as conn:
            numbers = [conn.get(int(i))['numbers'] for i in idx]
        return numbers

    def get_number_of_atoms(self, idx):
        if type(idx) is int:
            idx = np.array([idx])

        idx = idx + 1

        with connect(self.asedb) as conn:
            n_atoms = [conn.get(int(i))['natoms'] for i in idx]
        n_atoms = np.array(n_atoms)
        return n_atoms


class DataProvider:
    def __init__(self, data_reader, batch_size,
                 indices=None, shuffle=True, capacity=5000):
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._is_running = False

        if indices is None:
            self.indices = list(range(len(self.data_reader)))
        else:
            self.indices = indices

        # setup queue
        self.names = data_reader.names
        self.shapes = data_reader.shapes
        self.shapes = [self.shapes[n] for n in self.names]
        self.dtypes = data_reader.dtypes
        self.dtypes = [self.dtypes[n] for n in self.names]
        self.placeholders = {
            name: tf.placeholder(dt, shape=shape, name=name)
            for dt, name, shape in zip(self.dtypes, self.names, self.shapes)
        }

        self.queue = tf.PaddingFIFOQueue(capacity,
                                         dtypes=self.dtypes,
                                         shapes=self.shapes,
                                         names=self.names)
        self.enqueue_op = self.queue.enqueue(self.placeholders)
        self.dequeue_op = self.queue.dequeue()

    def __len__(self):
        return len(self.indices)

    def create_threads(self, sess, coord=None, daemon=False, start=True):
        if coord is None:
            coord = tf.train.Coordinator()

        if self._is_running:
            return []

        thread = threading.Thread(target=self._run, args=(sess, coord))

        if daemon:
            thread.daemon = True
        if start:
            thread.start()

        self._is_running = True
        return [thread]

    def _run(self, sess, coord=None):
        while not coord.should_stop():
            if self.shuffle:
                shuffle(self.indices)

            for bstart in range(0, len(self.indices) - self.batch_size + 1,
                                self.batch_size):
                batch = self.data_reader[
                    self.indices[bstart:bstart + self.batch_size]]
                feed_dict = {
                    self.placeholders[name]: batch[name]
                    for name in self.names
                }
                try:
                    sess.run(self.enqueue_op, feed_dict=feed_dict)
                except Exception as e:
                    coord.request_stop(e)

    def get_batch(self):
        return self.dequeue_op


def get_atoms_input(data):
    atoms_input = (
        data['numbers'], data['positions'], data['offset'], data['idx_ik'],
        data['idx_jk'], data['idx_j'], data['seg_m'], data['seg_i'],
        data['seg_j'], data['ratio_j']
    )
    return atoms_input
