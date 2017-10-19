import pytest
import numpy as np

from ase import Atoms
from ase.db import connect

import schnet

from .test_data import mol_path, atomref


def test_stats_per_atom(mol_path):
    reader = schnet.ASEReader(mol_path, ['energy_U0'], [], [])

    with connect(mol_path) as conn:
        # check length
        assert conn.count() == len(reader)

        # check batch indexing
        idx = np.random.randint(low=0, high=19, size=(10,))

        E = reader.get_property('energy_U0', idx)
        Z = reader.get_atomic_numbers(idx)
        N = reader.get_number_of_atoms(idx)

    EpA = E.ravel() / N.ravel()
    assert EpA.shape == (len(idx),)

    mu, std = schnet.stats_per_atom(E, Z, False)
    assert mu == np.mean(EpA)
    assert std == np.std(EpA)


def test_stats_per_atom_atomref(mol_path, atomref):
    reader = schnet.ASEReader(mol_path, ['energy_U0'], [], [])

    with connect(mol_path) as conn:
        # check length
        assert conn.count() == len(reader)

        # check batch indexing
        idx = np.arange(20)

        E = reader.get_property('energy_U0', idx).ravel()
        Z = reader.get_atomic_numbers(idx)
        N = reader.get_number_of_atoms(idx)

    E0 = np.array([np.sum(atomref[z]) for z in Z]).ravel()
    EpA = (E - E0) / N.ravel()
    assert EpA.shape == (len(idx),)

    mu, std = schnet.stats_per_atom(E, Z, False, atom_ref=atomref)
    assert mu == np.mean(EpA)
    assert std == np.std(EpA)
