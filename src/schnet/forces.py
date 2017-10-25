import numpy as np
import tensorflow as tf
from schnet.data import get_atoms_input

from scripts.train_energy_force import args


def predict_energy_forces(model, data):
    atoms_input = get_atoms_input(data)
    Ep = model(*atoms_input)
    Fp = -tf.convert_to_tensor(
        tf.gradients(tf.reduce_sum(Ep), atoms_input[1])[0])
    return Ep, Fp


def calculate_errors(Ep, Fp, data, energy_prop, force_prop,
                     rho, fit_energy=True, fit_forces=True):
    loss = 0.

    if args.forces != 'none':
        F = data[force_prop]
        fdiff = F - Fp
    else:
        fdiff = Fp

    fmse = tf.reduce_mean(fdiff ** 2)
    fmae = tf.reduce_mean(tf.abs(fdiff))

    if fit_forces:
        loss += tf.nn.l2_loss(fdiff)

    E = data[energy_prop]
    ediff = E - Ep
    eloss = tf.nn.l2_loss(ediff)

    emse = tf.reduce_mean(ediff ** 2)
    emae = tf.reduce_mean(tf.abs(ediff))

    if fit_energy:
        loss += rho * eloss

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