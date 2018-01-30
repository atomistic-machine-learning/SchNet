import os

import numpy as np
import tensorflow as tf
from time import time


class EarlyStopping:
    def __init__(self, output_dir, model, train_op,
                 train_loss, train_errors, val_loss, val_errors, summary_fn,
                 validation_batches, global_step,
                 save_interval=1000,
                 validation_interval=1000, summary_interval=1000):
        self.output_dir = output_dir
        self.model = model
        self.train_op = train_op
        self.train_loss = train_loss
        self.train_errors = train_errors
        self.val_loss = val_loss
        self.val_errors = val_errors
        self.summary_fn = summary_fn
        self.global_step = global_step
        self.validation_batches = validation_batches
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        self.summary_interval = summary_interval

        self.chkpt_dir = os.path.join(output_dir, 'chkpoints')
        self.chkpts = os.path.join(output_dir, 'chkpoints', 'chkpoints')
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'validation')
        self.models_dir = os.path.join(output_dir, 'validation/best_model')
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

        # validation loss
        self.loss_path = os.path.join(self.train_dir, 'loss.npz')
        if os.path.exists(self.loss_path):
            lf = np.load(self.loss_path)
            self.best_loss = lf['loss'].item()
            self.best_step = lf['step'].item()
        else:
            self.best_loss = np.Inf
            self.best_step = 0.
            np.savez(self.loss_path, loss=self.best_loss, step=self.best_step)

        self.saver = tf.train.Saver()
        self.val_writer = tf.summary.FileWriter(self.val_dir)
        self.start_iter = 0

    def train(self, sess, coord, max_steps):
        self.train_writer = tf.summary.FileWriter(self.train_dir,
                                                  graph=sess.graph)

        # restore
        chkpt = tf.train.latest_checkpoint(self.chkpt_dir)
        if chkpt is not None:
            self.start_iter = int(chkpt.split('-')[-1])
        rst_iter = self.global_step.assign(self.start_iter)

        if chkpt is not None:
            self.saver.restore(sess, chkpt)
            sess.run(rst_iter)

        step = self.start_iter
        while not coord.should_stop():
            if step > max_steps:
                coord.request_stop()
                break

            if step % self.summary_interval == 0:
                result = sess.run(
                    [self.train_op, self.global_step,
                     self.train_loss] + self.train_errors
                )
                step = result[1]
                loss = result[2]
                errors = result[3:]
                _, summary = self.summary_fn(loss, list(zip(*[errors])))
                self.train_writer.add_summary(summary, global_step=step)
            else:
                start = time()
                _, step = sess.run([self.train_op, self.global_step])
                print('Train step:', time() - start)

            if step % self.validation_interval == 0:
                loss = []
                errors = []
                for i in range(self.validation_batches):
                    result = sess.run(
                        [self.val_loss] + self.val_errors
                    )
                    loss.append(result[0])
                    errors.append(result[1:])

                vloss, summary = self.summary_fn(loss, list(zip(*errors)))
                self.val_writer.add_summary(summary, global_step=step)
                if vloss < self.best_loss:
                    self.best_loss = vloss
                    self.best_step = step
                    np.savez(self.loss_path, loss=self.best_loss,
                             step=self.best_step)
                    self.model.save(sess, self.models_dir, global_step=step)

            if step % self.save_interval == 0:
                self.saver.save(sess, self.chkpts, global_step=step)


def build_train_op(loss, optimizer, global_step, moving_avg_decay=0.99,
                   dependencies=[]):
    grads = optimizer.compute_gradients(loss)
    apply_gradient_op = optimizer.apply_gradients(grads,
                                                  global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_avg_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(
            [apply_gradient_op,
             variables_averages_op] + update_ops + dependencies):
        train_op = tf.no_op(name='train')
    return train_op
