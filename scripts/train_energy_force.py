import argparse
import logging
import os
from functools import partial

import numpy as np
from schnet.nn.train import EarlyStopping, build_train_op
import tensorflow as tf
from schnet.atoms import stats_per_atom
from schnet.data import ASEReader, DataProvider
from schnet.models import SchNet

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_schnet_input(data):
    schnet_input = (
        data['numbers'], data['positions'], data['offset'], data['idx_ik'],
        data['idx_jk'], data['idx_j'], data['seg_m'], data['seg_i'],
        data['seg_j'], data['ratio_j']
    )
    return schnet_input


def predict_energy_forces(schnet, data):
    schnet_input = get_schnet_input(data)
    Ep = schnet(*schnet_input)
    Fp = -tf.convert_to_tensor(
        tf.gradients(tf.reduce_sum(Ep), schnet_input[1])[0])
    return Ep, Fp


def calculate_errors(args, data, Ep, Fp):
    loss = 0.

    if args.forces != 'none':
        F = data[args.forces]
        fdiff = F - Fp
    else:
        fdiff = Fp

    fmse = tf.reduce_mean(fdiff ** 2)
    fmae = tf.reduce_mean(tf.abs(fdiff))

    if args.fit_force:
        loss += tf.nn.l2_loss(fdiff)

    E = data[args.energy]
    ediff = E - Ep
    eloss = tf.nn.l2_loss(ediff)

    emse = tf.reduce_mean(ediff ** 2)
    emae = tf.reduce_mean(tf.abs(ediff))

    if args.fit_energy:
        loss += args.eweight * eloss

    errors = [emse, emae, fmse, fmae]
    return loss, errors


def collect_summaries(args, loss, errors):
    emse, emae, fmse, fmae = errors
    vloss = np.sum(loss)

    summary = tf.Summary()
    summary.value.add(tag='loss', simple_value=vloss)
    summary.value.add(tag='total_energy_RMSE',
                      simple_value=np.sqrt(np.mean(emse[0])))
    summary.value.add(tag='total_energy_MAE', simple_value=np.mean(emae))
    if args.forces != 'none':
        summary.value.add(tag='force_RMSE', simple_value=np.sqrt(np.mean(fmse)))
        summary.value.add(tag='force_MAE', simple_value=np.mean(fmae))
    return vloss, summary


def train(args):
    logging.info('Setup directories')
    splitpath = args.splits
    if not splitpath.endswith('.npz'):
        splitpath += '.npz'
    splitname = splitpath.split('/')[-1].split('.')[0]

    # initialize output files
    name = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}'.format(
        'SchNet', args.fit_energy, args.fit_force,
        args.basis, args.filters, args.interactions, args.cutoff,
        args.lr, args.eweight, splitname
    )
    if len(args.name) > 0:
        name += '_' + args.name

    outdir = os.path.join(args.output_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    args_log = os.path.join(outdir, 'args.npy')
    np.save(args_log, args)

    # setup data pipeline
    logging.info('Setup data reader')
    forces = [args.forces] if args.forces != 'none' else []
    data_reader = ASEReader(args.data,
                            [args.energy],
                            forces, [(None, 3)])

    if not os.path.exists(splitpath):
        logging.info('Create data splits')
        if args.ntrain < 0 and args.nval < 0:
            logging.error('If `splits` does not exist, arguments ' +
                          '`ntrain` and `nval` have to be specified!')
            return

        N = len(data_reader)
        pidx = np.random.permutation(N)
        train_idx = pidx[:args.ntrain]
        val_idx = pidx[args.ntrain:args.ntrain + args.nval]
        test_idx = pidx[args.ntrain + args.nval:]
        np.savez(splitpath, train_idx=train_idx,
                 val_idx=val_idx, test_idx=test_idx)
    else:
        logging.info('Load existing data splits')
        S = np.load(splitpath)
        train_idx = S['train_idx']
        val_idx = S['val_idx']

    atomref = None
    try:
        atomref = np.load(args.atomref)['atom_ref']
        if args.energy == 'energy_U0':
            atomref = atomref[:, 1:2]
        if args.energy == 'energy_U':
            atomref = atomref[:, 2:3]
        if args.energy == 'enthalpy_H':
            atomref = atomref[:, 3:4]
        if args.energy == 'free_G':
            atomref = atomref[:, 4:5]
        if args.energy == 'Cv':
            atomref = atomref[:, 5:6]
    except Exception as e:
        print(e)

    logging.info('Setup train/validation data providers')
    train_provider = DataProvider(data_reader, args.batch_size,
                                  train_idx)
    train_data = train_provider.get_batch()
    val_provider = DataProvider(data_reader, args.valbatch, val_idx)
    val_data = val_provider.get_batch()

    logging.info('Collect train data statistics')
    E = data_reader.get_property(args.energy, train_idx)
    Z = data_reader.get_atomic_numbers(train_idx)
    mean_energy_per_atom, stddev_energy_per_atom = \
        stats_per_atom(E, Z, args.intensive, atomref)
    logging.info('Energy statistics: mu/atom=' + str(mean_energy_per_atom) +
                 ', std/atom=' + str(stddev_energy_per_atom))

    logging.info('Setup model')
    schnet = SchNet(args.interactions, args.basis, args.filters, args.cutoff,
                    mean_energy_per_atom, stddev_energy_per_atom,
                    atomref=atomref, intensive=args.intensive,
                    filter_pool_mode=args.filter_pool_mode)

    # apply model
    Ep_train, Fp_train = predict_energy_forces(schnet, train_data)
    Ep_val, Fp_val = predict_energy_forces(schnet, val_data)

    # calculate loss, errors & summaries
    train_loss, train_errors = calculate_errors(args, train_data, Ep_train,
                                                Fp_train)
    val_loss, val_errors = calculate_errors(args, val_data, Ep_val, Fp_val)

    # setup training
    logging.info('Setup optimizer')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(args.lr, global_step, 100000, 0.96)

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = build_train_op(train_loss, optimizer, global_step)

    logging.info('Setup early stopping')
    valbatches = len(val_provider) // args.valbatch
    trainer = EarlyStopping(outdir, schnet, train_op,
                            train_loss, train_errors,
                            val_loss, val_errors,
                            partial(collect_summaries, args),
                            validation_batches=valbatches,
                            global_step=global_step)

    with tf.Session() as sess:
        logging.info('Start queue runners')
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        train_provider.create_threads(sess, coord)
        val_provider.create_threads(sess, coord)

        logging.info('Start training')
        trainer.train(sess, coord, args.max_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        help='Path to training data (ASE DB)')
    parser.add_argument('output_dir',
                        help='Output directory for model and training log.')
    parser.add_argument('splits',
                        help='Path to data splits (is created, if not existing)')
    parser.add_argument('--ntrain', help='Number of training examples',
                        type=int, default=-1)
    parser.add_argument('--nval', help='Number of validation examples',
                        type=int, default=-1)
    parser.add_argument('--energy', help='Name of run',
                        default='energy_U0')
    parser.add_argument('--forces', help='Name of run',
                        default='none')
    parser.add_argument('--filter_pool_mode', help='One out of [sum, mean]',
                        default='sum')
    parser.add_argument('--intensive', action='store_true',
                        help='If intensive, mean pool energy over atoms')
    parser.add_argument('--fit_energy', action='store_true',
                        help='Use additional energy loss')
    parser.add_argument('--fit_force', action='store_true',
                        help='Use additional energy loss')
    parser.add_argument('--atomref', help='Atom reference file (NPZ)',
                        default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size',
                        default=32)
    parser.add_argument('--cutoff', type=float, help='Distance cutoff',
                        default=20.)
    parser.add_argument('--interactions', type=int, help='Distance cutoff',
                        default=6)
    parser.add_argument('--basis', type=int, help='Basis set size',
                        default=64)
    parser.add_argument('--filters', type=int, help='Factor space size',
                        default=64)
    parser.add_argument('--eweight', type=float,
                        help='Number of steps to saturate eloss scale',
                        default=1.)
    parser.add_argument('--max_steps', type=int, help='Number of steps',
                        default=5000000)
    parser.add_argument('--valint', type=int, help='Validation interval',
                        default=5000)
    parser.add_argument('--valbatch', type=int,
                        help='Size of validation batches',
                        default=100)
    parser.add_argument('--name', help='Name of run',
                        default='')
    parser.add_argument('--lr', type=float, help='Learning rate',
                        default=1e-3)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)
