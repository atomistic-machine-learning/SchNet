from functools import wraps

import tensorflow as tf


class MetaModule(type):
    def __new__(cls, name, bases, local):
        cls = type.__new__(cls, name, bases, local)

        def store_init_args(fn):
            @wraps(fn)
            def store_init_args_wrapper(self, *args, **kwargs):
                fn(self, *args, **kwargs)
                self._args = args
                self._kwargs = kwargs

            return store_init_args_wrapper

        cls.__init__ = store_init_args(cls.__init__)
        return cls


class Module(metaclass=MetaModule):
    """ Base class for all NN layers"""

    def __init__(self, name=None):
        self._name = self.__class__.__name__ if name is None else name
        self._parent = None
        self._children = []
        with tf.variable_scope(None, default_name=self._name) as scope:
            self._scope = scope.name
            self._initialize()
        if len(self.variables.keys()) > 0:
            self.saver = tf.train.Saver(self.variables,
                                        save_relative_paths=True)
        else:
            self.saver = None

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self._scope, reuse=True):
            res = self._forward(*args, **kwargs)
        return res

    def _initialize(self):
        ''' Initialize all TF variable and sub-layers here '''
        pass

    def _forward(self, *args, **kwargs):
        ''' Implement forward pass here. No new variable allowed! '''
        raise NotImplementedError

        # def signal(self, *args, **kwargs):
        #     ''' Implement signal estimator here '''
        #     raise NotImplementedError

    def save(self, sess, save_path, global_step=None):
        if self.saver is None:
            print('No variables to save!')
        else:
            self.saver.save(sess, save_path, global_step)

    def restore(self, sess, save_path):
        if self.saver is None:
            print('No variables to restore!')
        else:
            self.saver.restore(sess, save_path)

    def create_child(self):
        print(*self._args, **self._kwargs)
        child = type(self)(*self._args, **self._kwargs)
        self._children.append(child)
        child._parent = self
        child._pullops = child._create_pullops()
        return child

    @property
    def variables(self):
        scope_filter = self._scope + '/'
        varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=scope_filter)
        variables = {v.name[len(scope_filter):]: v for v in varlist}
        return variables

    def _create_pullops(self):
        child_vars = self.variables
        parent_vars = self._parent.variables
        pull_ops = []
        for vname in child_vars.keys():
            pull_op = tf.assign(child_vars[vname], parent_vars[vname])
            pull_ops.append(pull_op)
        return pull_ops

    def pull(self, session):
        assert self._parent, 'Pull can only be used for child layers!'
        session.run(self._pullops)
