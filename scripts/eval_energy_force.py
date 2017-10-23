import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from ase.db import connect

from schnet.schnet import SchNet

batch_size = 100


def ase_iterator(path, indices, energy_name, forces_name):
    with connect(path) as conn:
        for i in indices:
            row = conn.get(i+1)
            at = row.toatoms()
            energy = row[energy_name]
            if forces_name != 'none':
                forces = row.data[forces_name]
            else:
                forces = np.zeros_like(at.positions)
            yield row.id, at, energy, forces


def prepare_input(at):
    features = {
        'numbers': np.array(at.numbers, dtype=np.int64).reshape(1, -1),
        'positions': np.array(at.positions, dtype=np.float32).reshape(1, -1, 3)
    }
    return features


def eval(model_path, data_path, energy, forces, name):
    tf.reset_default_graph()
    checkpoint_dir = os.path.join(model_path, 'val')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    print(ckpt)

    args = np.load(
        os.path.join(model_path,
                     'args.npy')).item()

    # load reference energies
    atomref = None
    try:
        atomref = np.load(args.atom_refs)['atom_ref'][:, 1:2]
    except Exception as e:
        print(e)

    data = {
        'numbers': tf.placeholder(tf.int64, shape=(None, None)),
        'positions': tf.placeholder(tf.float32, shape=(None, None, 3))
    }

    # initialize model
    normalized_filters = None
    try:
        normalized_filters = args.normalized_filters
    except Exception:
        pass

    schnet = SchNet(args.interactions, args.basis, args.filters, args.cutoff,
                    shared_interactions=args.shared_interactions,
                    normalized_filters=normalized_filters,
                    atomref=atomref)

    # apply model
    Z = data['numbers']
    R = data['positions']
    Ep = schnet(Z, R, keep_prob=1.)
    Fp = -tf.gradients(tf.reduce_sum(Ep), R)[0]

    aids = []
    Epred = []
    Fpred = []
    E = []
    F = []
    count = 0
    with tf.Session() as sess:
        schnet.restore(sess, ckpt)
        for id, at, e, f in ase_iterator(data_path, energy, forces):
            features = prepare_input(at)
            feed = {v: features[k] for k, v in data.items()}
            ep, fp = sess.run([Ep, Fp], feed_dict=feed)
            Epred.append(ep)
            Fpred.append(fp)
            E.append(e)
            F.append(f)
            aids.append(id)

            count += 1
            if count % 1000 == 0:
                print(count)

    E = np.array(E)
    Epred = np.array(Epred)
    aids = np.array(aids)
    e_mae = np.mean(np.abs(E - Epred[:, 0, 0]))
    e_rmse = np.sqrt(np.mean(np.square(E - Epred[:, 0, 0])))

    if args.forces != 'none':
        F = np.array(F)
        Fpred = np.array(Fpred)
        f_mae = np.mean(np.abs(F - Fpred[:, 0]))
        f_rmse = np.sqrt(np.mean(np.square(F - Fpred[:, 0])))
    else:
        F = None
        Fpred = None
        f_mae = 0.
        f_rmse = 0.
    np.savez(os.path.join(model_path, 'results_' + name + '.npz'),
             aid=aids, F=F, Fpred=Fpred, E=E, Epred=Epred)
    return e_mae, e_rmse, f_mae, f_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to directory with models')
    parser.add_argument('data',
                        help='Path to data splits')
    parser.add_argument('split',
                        help='train / val / test')
    parser.add_argument('--energy', help='Name of run',
                        default='energy_U0')
    parser.add_argument('--forces', help='Name of run',
                        default='none')
    args = parser.parse_args()

    with open(os.path.join(args.path, 'errors_'+args.split+'.csv'), 'w') as f:
        f.write('model,energy MAE,energy RMSE,force MAE,force RMSE\n')
        for dir in os.listdir(args.path):
            mdir = os.path.join(args.path, dir)
            if not os.path.isdir(mdir):
                continue
            split_name = '_'.join(dir.split('_')[-2:])
            data_path = os.path.join(args.data, split_name, args.split+'.db')
            res = eval(mdir, data_path,
                       args.energy, args.forces, args.split)
            res = [str(np.round(r, 4)) for r in res]
            f.write(dir + ',' + ','.join(res) + '\n')
            print(dir, res)
