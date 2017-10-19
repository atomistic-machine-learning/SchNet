import argparse
import logging
import os
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from urllib.error import URLError, HTTPError

import numpy as np
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Hartree, eV, Bohr, Ang

from schnet.atoms import collect_neighbors


def load_atomrefs(at_path):
    logging.info('Downloading GDB-9 atom references...')
    at_url = 'https://ndownloader.figshare.com/files/3195395'
    tmpdir = tempfile.mkdtemp('gdb9')
    tmp_path = os.path.join(tmpdir, 'atomrefs.txt')

    try:
        urllib.request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")
    except HTTPError as e:
        logging.error("HTTP Error:", e.code, at_url)
        return False
    except URLError as e:
        logging.error("URL Error:", e.reason, at_url)
        return False

    atref = np.zeros((100, 6))
    labels = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
    with open(tmp_path) as f:
        lines = f.readlines()
        for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
            atref[z, 0] = float(l.split()[1])
            atref[z, 1] = float(l.split()[2]) * Hartree / eV
            atref[z, 2] = float(l.split()[3]) * Hartree / eV
            atref[z, 3] = float(l.split()[4]) * Hartree / eV
            atref[z, 4] = float(l.split()[5]) * Hartree / eV
            atref[z, 5] = float(l.split()[6])
    np.savez(at_path, atom_ref=atref, labels=labels)
    return True


def load_data(dbpath):
    logging.info('Downloading GDB-9 data...')
    tmpdir = tempfile.mkdtemp('gdb9')
    tar_path = os.path.join(tmpdir, 'gdb9.tar.gz')
    raw_path = os.path.join(tmpdir, 'gdb9_xyz')
    url = 'https://ndownloader.figshare.com/files/3195389'

    try:
        urllib.request.urlretrieve(url, tar_path)
        logging.info("Done.")
    except HTTPError as e:
        logging.error("HTTP Error:", e.code, url)
        return False
    except URLError as e:
        logging.error("URL Error:", e.reason, url)
        return False

    tar = tarfile.open(tar_path)
    tar.extractall(raw_path)
    tar.close()

    prop_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve', 'energy_U0', 'energy_U', 'enthalpy_H',
                  'free_G', 'Cv']
    conversions = [1., 1., 1., 1., Bohr ** 3 / Ang ** 3,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Bohr ** 2 / Ang ** 2, Hartree / eV,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Hartree / eV, 1.]

    logging.info('Parse xyz files...')
    with connect(dbpath) as con:
        for i, xyzfile in enumerate(os.listdir(raw_path)):
            xyzfile = os.path.join(raw_path, xyzfile)

            if i % 10000 == 0:
                logging.info('Parsed: ' + str(i) + ' / 133885')
            properties = {}
            tmp = os.path.join(tmpdir, 'tmp.xyz')

            with open(xyzfile, 'r') as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p, c in zip(prop_names, l, conversions):
                    properties[pn] = float(p) * c
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace('*^', 'e'))

            with open(tmp, 'r') as f:
                ats = list(read_xyz(f, 0))[0]

            idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = \
                collect_neighbors(ats, 20.)

            data = {'_idx_ik': idx_ik, '_idx_jk': idx_jk, '_idx_j': idx_j,
                    '_seg_i': seg_i, '_seg_j': seg_j, '_offset': offset,
                    '_ratio_j': ratio_j}
            con.write(ats, key_value_pairs=properties, data=data)
    logging.info('Done.')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destpath', help='Path to QM9 directory')
    args = parser.parse_args()

    if not os.path.exists(args.destpath):
        os.makedirs(args.destpath)

    load_atomrefs(os.path.join(args.destpath, 'atomref.npz'))
    load_data(os.path.join(args.destpath, 'qm9.db'))
