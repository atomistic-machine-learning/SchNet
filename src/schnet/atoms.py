import numpy as np
from ase.neighborlist import NeighborList


class IsolatedAtomException(Exception):
    pass


def stats_per_atom(data, numbers, is_per_atom, atom_ref=None):
    Y = np.array(data)
    Y = Y.reshape((-1, 1))
    natoms = np.vstack([len(z) for z in numbers]).reshape((-1, 1))

    if atom_ref is not None:
        Y0 = np.vstack(
            [np.sum(atom_ref[np.array(z)], 0) for z in numbers]).reshape(
            (-1, 1))
        if is_per_atom:
            Y0 /= natoms.reshape((-1, 1))
        Y -= Y0

    if not is_per_atom:
        Y /= natoms.reshape((-1, 1))

    mu = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)
    return mu, std


def collect_neighbors(at, cutoff):
    '''
      Collect the neighborhood indices for an atomistic system, respecting
      periodic boundary conditions.

    :param ase.Atoms at: atomistic system
    :param float cutoff: neighborhood cutoff
    :return: neighborhood index vectors and offsets in lattice coordinates
    '''
    nbhlist = NeighborList(cutoffs=[cutoff*0.5] * len(at), bothways=True,
                           self_interaction=False)
    nbhlist.build(at)
    cell = at.cell

    idx_ik = []
    seg_i = []
    idx_j = []
    seg_j = []
    idx_jk = []
    offset = []
    ratio_j = []
    c_sites = 0
    for i in range(len(at)):
        ind, off = nbhlist.get_neighbors(i)
        sidx = np.argsort(ind)
        ind = ind[sidx]
        off = np.dot(off[sidx], cell)
        uind = np.unique(ind)

        idx_ik.append([i] * len(ind))
        seg_i.append([i] * len(uind))
        idx_j.append(uind)
        idx_jk.append(ind)
        offset.append(off)
        if len(ind) > 0:
            tmp = np.nonzero(np.diff(np.hstack((-1, ind, np.Inf))))[0]
            rep = np.diff(tmp)
            ratio_j.append(rep / np.sum(rep))
            seg_ij = np.repeat(np.arange(len(uind)), rep) + c_sites
            seg_j.append(seg_ij)
            c_sites = seg_ij[-1] + 1
        else:
            print(at)
            raise IsolatedAtomException

    seg_i = np.hstack(seg_i)
    if len(seg_j) > 0:
        seg_j = np.hstack(seg_j)
    else:
        seg_j = np.array([])
    idx_ik = np.hstack(idx_ik)
    idx_j = np.hstack(idx_j)
    idx_jk = np.hstack(idx_jk)
    ratio_j = np.hstack(ratio_j)

    offset = np.vstack(offset).astype(np.float32)
    return idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j
