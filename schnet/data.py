import logging
import threading
from random import shuffle

import numpy as np
import tensorflow as tf
from ase.db import connect
from ase.neighborlist import NeighborList


class IsolatedAtomException(Exception):
    pass


def collect_neighbors(at, cutoff: float):
    '''
      Collect the neighborhood indices for an atomistic system, respecting
      periodic boundary conditions.

    :param ase.Atoms at: atomistic system
    :param float cutoff: neighborhood cutoff
    :return: neighborhood index vectors and offsets in lattice coordinates
    '''
    nbhlist = NeighborList(cutoffs=[cutoff] * len(at), bothways=True,
                           self_interaction=False)
    nbhlist.build(at)

    idx_ik = []
    seg_i = []
    idx_j = []
    seg_j = []
    idx_jk = []
    offset = []
    c_sites = 0
    for i in range(len(at)):
        ind, off = nbhlist.get_neighbors(i)
        sidx = np.argsort(ind)
        ind = ind[sidx]
        off = off[sidx]
        uind = np.unique(ind)

        idx_ik.append([i] * len(ind))
        seg_i.append([i] * len(uind))
        idx_j.append(uind)
        idx_jk.append(ind)
        offset.append(off)
        if len(ind) > 0:
            tmp = np.nonzero(np.diff(np.hstack((-1, ind, np.Inf))))[0]
            rep = np.diff(tmp)
            seg_ij = np.repeat(np.arange(len(uind)), rep) + c_sites
            seg_j.append(seg_ij)
            c_sites = seg_ij[-1] + 1
        else:
            raise IsolatedAtomException

    seg_i = np.hstack(seg_i)
    if len(seg_j) > 0:
        seg_j = np.hstack(seg_j)
    else:
        seg_j = np.array([])
    idx_ik = np.hstack(idx_ik)
    idx_j = np.hstack(idx_j)
    idx_jk = np.hstack(idx_jk)

    offset = np.vstack(offset).astype(np.float32)
    return idx_ik, seg_i, idx_j, idx_jk, seg_j, offset


def generate_neighbor_dataset(asedb, nbhdb, cutoff):
    '''
    Generates an ASE DB with neighborhood information from an ASE DB without it.

    :param str asedb: path to original data
    :param str nbhdb: destination
    :param float cutoff: neighborhood cutoff radius
    '''
    with connect(asedb, use_lock_file=False) as srccon:
        with connect(nbhdb, use_lock_file=False) as dstcon:
            for row in srccon.select():
                at = row.toatoms()
                res = collect_neighbors(at, cutoff)
                if res is None:
                    logging.warning('Skipping example ' + str(row.id) +
                                    ' due to isolated atom with r_cut=' +
                                    str(cutoff))
                    continue
                idx_ik, seg_i, idx_j, idx_jk, seg_j, offset = res
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

    def __init__(self, asedb, kvp_properties, data_properties, data_shapes,
                 preload=False, subset=None):
        self.asedb = asedb
        self.kvp = {k: (None, 1) for k in kvp_properties}
        self.data_props = dict(zip(data_properties, data_shapes))
        self.properties = list(self.kvp.keys())+list(self.data_props.keys())
        self.subset = subset
        self.preload = preload

        if preload:
            subset = range(len(self)) if subset is None else subset
            self.data = []
            with connect(self.asedb) as conn:
                for i in subset:
                    self.data.append(conn.get(i))
        else:
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
            'charges': (None,),
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

        if self.subset is not None and not self.preload:
            idx = self.subset[idx]

        if self.preload:
            data = self._build_batch(self.data, idx)
        else:
            with connect(self.asedb) as conn:
                data = self._build_batch(conn, idx)

        data = {p: np.concatenate(b, axis=0) for p, b in data.items()}
        return data

    def _load_row(self, data, idx):
        if self.preload:
            return data[idx]
        else:
            return data.get(idx)

    def _build_batch(self, datasrc, idx):
        data = {
            'seg_m': [],
            'idx_ik': [],
            'seg_i': [],
            'idx_j': [],
            'idx_jk': [],
            'seg_j': [],
            'charges': [],
            'positions': [],
            'offset': [],
            'cells': []
        }

        for prop in self.properties:
            data[prop] = []

        c_atoms = 0
        c_sites = 0
        c_site_segs = 0
        for k, i in enumerate(idx):
            row = self._load_row(datasrc, i)
            at = row.toatoms()

            if len(row.data['_idx_jk']) > 0:
                upd_sites = row.data['_idx_jk'][-1] + 1
            else:
                upd_sites = 0

            if len(row.data['_seg_j']) > 0:
                upd_site_segs = row.data['_seg_j'][-1] + 1
            else:
                upd_site_segs = 0

            data['seg_m'].append(np.array([k] * at.get_number_of_atoms()))
            data['idx_ik'].append(row.data['_idx_ik'] + c_atoms)
            data['seg_i'].append(row.data['_seg_i'] + c_atoms)
            data['idx_j'].append(row.data['_idx_j'] + c_atoms)
            data['idx_jk'].append(row.data['_idx_jk'] + c_sites)
            data['seg_j'].append(row.data['_seg_j'] + c_site_segs)
            data['offset'].append(row.data['_offset'])
            data['charges'].append(at.get_atomic_numbers())
            data['positions'].append(at.get_positions())
            data['cells'].append([at.cell])
            c_atoms += at.get_number_of_atoms()
            c_sites += upd_sites
            c_site_segs += upd_site_segs

            for prop in self.kvp.keys():
                data[prop].append(np.array([[row[prop]]]))

            for prop in self.data_props.keys():
                data[prop].append(row.data[prop])
        return data


class DataProvider:
    def __init__(self, data_reader, batch_size,
                 capacity=5000,
                 indices=None, shuffle=True):
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
            name: tf.placeholder(dt, name=name)
            for dt, name in zip(self.dtypes, self.names)
        }
        print(len(self.names), len(self.dtypes), len(self.shapes))
        print(len(self.placeholders))
        print(self.names, self.dtypes)

        self.queue = tf.PaddingFIFOQueue(capacity,
                                         dtypes=self.dtypes,
                                         shapes=self.shapes,
                                         names=self.names)
        self.enqueue_op = self.queue.enqueue(self.placeholders)
        self.dequeue_op = self.queue.dequeue()

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
        # feat_dict = {}
        # for name, feat, shape in zip(self.names, self.dequeue_op,
        #                              self.shapes):
        #     feat_dict[name] = feat
        return self.dequeue_op
