from pathlib import Path
from types import SimpleNamespace
import json
import warnings

import numpy as np

from .. import ssnode
from ..utils import cached_property
from .loader import LazyLogFileLoader


class MomentMatchingData(object):

    run_module = 'bptt_moments'

    def __init__(self, datastore):
        self.datastore = Path(datastore)
        if not self.datastore.is_dir():
            self.datastore = self.datastore.parent
        self.log = LazyLogFileLoader(
            self.datastore,
            ['learning.csv', 'generator.csv', 'gen_moments.csv'])

        with open(self.datastore.joinpath('info.json')) as file:
            self.info = json.load(file)

    @property
    def total_steps(self):
        return min(
            len(self.log.learning.data),
            len(self.log.generator.data),
            len(self.log.gen_moments.data),
        )

    def __make_data_getter(name):
        @property
        def getter(self):
            data = getattr(self.log, name).data
            if len(data) > self.total_steps:
                warnings.warn('Only using first {} rows from {}.csv'
                              ' since it is longer than others.'
                              .format(self.total_steps, name))

            return data[:self.total_steps]
        return getter

    learning = __make_data_getter('learning')
    generator = __make_data_getter('generator')
    gen_moments_raw = __make_data_getter('gen_moments')

    @property
    def epochs(self):
        return self.step_to_epoch(self.learning[:, 0])

    def step_to_epoch(self, step):
        truth_size = self.info['run_config']['truth_size']  # data size
        batchsize = self.info['run_config']['batchsize']
        return step * batchsize / truth_size

    def epoch_to_step(self, epoch):
        truth_size = self.info['run_config']['truth_size']  # data size
        batchsize = self.info['run_config']['batchsize']
        return epoch / batchsize * truth_size

    def true_param(self, name):
        return np.asarray(self.info['run_config']
                          .get('true_ssn_options', {})
                          .get(name, ssnode.DEFAULT_PARAMS[name]))

    def iter_gen_params(self, indices=None):
        if indices is None:
            indices = slice(None)
        gen = self.gen_matrices
        return zip(gen.J[indices], gen.D[indices], gen.S[indices])

    def gen_param(self, name):
        return getattr(self.gen_matrices, name)

    @cached_property
    def gen_matrices(self):
        gen = self.generator
        J, D, S = gen[:, 1:].reshape((-1, 3, 2, 2)).swapaxes(0, 1)
        return SimpleNamespace(J=J, D=D, S=S)

    @cached_property
    def gen_moments(self):
        data = self.gen_moments_raw
        return data.reshape((len(data), 2, -1))

    default_spec_keys = ('lam', 'learning_rate')

    def pretty_spec(self, keys):
        if keys is None:
            keys = self.default_spec_keys
        run_config = self.info['run_config']
        return ' '.join('{}={}'.format(k, run_config[k]) for k in keys)

    def params(self):
        return self.info['run_config']
