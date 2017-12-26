from pathlib import Path
import json
import warnings

import numpy as np
import pandas

from ..networks.ssn import concat_flat
from ..networks.utils import gridify_tc_data, sampled_tc_axes
from ..utils import cached_property
from .datastore_loader import get_datastore
from .run_configs import get_run_config, guess_run_module


def cached_record(name):
    def get(self):
        df = self.datastore.load(name)
        self.insert_epoch_column(df)
        return df
    get.__name__ = name
    return cached_property(get)


def element_to_array_names(element_names):
    """
    Convert a list of element names to a list of array names.

    >>> element_to_array_names(['X_E', 'X_I', 'Y_E', 'Y_I'])
    ['X', 'Y']

    """
    names = []
    for element in element_names:
        prefix, _ = element.split('_', 1)
        if prefix not in names:
            names.append(prefix)
    return names


def tc_samples_as_dataframe(data, rc=None, **kwargs):
    # The experiment parameters flatten in the second axis of `.truth`.
    names = sampled_tc_axes[1:]  # skip 'sample'
    keys = [n + 's' for n in names]  # turn them to plural

    if rc is None:
        values = [kwargs.pop(n) for n in keys]
    else:
        values = [getattr(rc, n) for n in keys]
    if kwargs:
        raise TypeError('Unrecognized arguments: {}'.format(sorted(kwargs)))

    columns = pandas.MultiIndex.from_product(values, names=names)
    df = pandas.DataFrame(data, columns=columns)
    df.index.name = sampled_tc_axes[0]

    return df


class BaseRecords(object):

    def __init__(self, datastore, info, rc):
        self.datastore = datastore
        self.info = info
        self.rc = rc

    @property
    def data_version(self):
        return self.info.get('extra_info', {}).get('data_version', 0)

    @property
    def run_module(self):
        return guess_run_module(self.info)

    learning = cached_record('learning')
    generator = cached_record('generator')
    truth = cached_property(lambda self: self.datastore.load('truth'))

    def insert_epoch_column(self, df):
        if 'disc_step' in df.columns:
            disc_updates = np.arange(1, len(df) + 1)
            df.loc[:, 'epoch'] = self.rc.disc_updates_to_epoch(disc_updates)
        elif 'gen_step' in df.columns:
            gen_step = df.loc[:, 'gen_step']
            df.loc[:, 'epoch'] = self.rc.gen_step_to_epoch(gen_step)

    @property
    def param_element_names(self):
        """
        Individual generator parameter names.

        It is a subset of `.generator.columns`.
        Typically, it is a list like ``['J_EE', 'J_EI', ...]``.

        At recording (run) time, these names are generated by the
        method `gen.get_flat_param_names
        <.TuningCurveGenerator.get_flat_param_names>` and recorded by
        `.FlexGenParamRecorder`.

        """
        return [n for n in self.generator.columns
                if n not in ('gen_step', 'epoch')]
    # See:
    # * 'gen_step' added in [[../recorders.py::gen_param_dtype]]
    # * 'epoch' added in [[def insert_epoch_column]]

    @property
    def param_array_names(self):
        """ Generator parameter array names; e.g., ``['J', 'D', 'S']``. """
        return element_to_array_names(self.param_element_names)

    def flatten_true_params(self, ndim=1):
        assert ndim in (1, 2)
        params = concat_flat(map(self.rc.get_true_param,
                                 self.param_array_names))
        params = np.asarray(params)
        if ndim == 2:
            return params.reshape((1, -1))
        return params

    def flatten_gen_params(self):
        return self.generator.loc[:, self.param_element_names].as_matrix()

    def truth_grid(self):
        """
        Accessing `.truth` as order-5 array.

        See: `gridify_tc_data`.
        """
        return gridify_tc_data(
            self.truth,
            num_contrasts=len(self.rc.contrasts),
            num_bandwidths=len(self.rc.bandwidths),
            num_cell_types=len(self.rc.cell_types),
            num_probes=len(self.rc.norm_probes),
        )

    def truth_df(self):
        """
        Accessing `.truth` as multi-index dataframe.
        """
        return tc_samples_as_dataframe(self.truth, rc=self.rc)

    @classmethod
    def from_info(cls, info, datastore_path):
        rc = get_run_config(info)
        datastore = get_datastore(datastore_path, info)
        if rc.is_legacy:
            # TODO: test loading legacy GAN data or ditch them completely.
            warnings.warn('Loading legacy GAN data is not well tested!')
        return cls(datastore, info, rc)

    def pretty_spec(self, keys=None):
        if keys is None:
            keys = []
        spec = ' '.join('{}={}'.format(k, getattr(self.rc, k)) for k in keys)
        return '{}: {}'.format(self.rc.gan_type, spec)


class GANRecords(BaseRecords):

    TC_mean = cached_record('TC_mean')
    disc_param_stats = cached_record('disc_param_stats')
    disc_learning = cached_record('disc_learning')

    @property
    def disc_param_stats_names(self):
        return [name for name in self.disc_param_stats.columns
                if name not in ('gen_step', 'disc_step', 'epoch')]


class ConditionalGANRecords(GANRecords):

    tc_stats = cached_record('tc_stats')


class MomentMatchingRecords(BaseRecords):

    gen_moments = cached_record('gen_moments')

    moment_names = ['mean', 'var']
    # it must be consistent with [[./datastore_loader.py::load_gen_moments]]
    # TODO: generate it from gen_moments or generate gen_moments from it.

    @cached_property
    def data_moments(self):
        """
        Data moments calculated from ``"truth.npy"``.

        To be consistent with `.gen_moments`, it is a `pandas.Series`
        instead of an array.  This works nicely with broadcasting in
        pandas, e.g., taking difference is as easy as
        ``self.gen_moments.sub(self.data_moments)``.
        """
        from ..networks.moment_matching import sample_moments
        array = sample_moments(self.truth)
        index = self.moment_names
        return pandas.DataFrame(array, index=index).stack()

    def insert_epoch_column(self, df):
        if 'gen_step' in df.columns:
            step = df.loc[:, 'gen_step']
        elif 'step' in df.columns:
            step = df.loc[:, 'step']
        else:
            return
        df.loc[:, 'epoch'] = self.rc.step_to_epoch(step)

    def pretty_spec(self, keys=None):
        if keys is None:
            keys = []
        spec = ' '.join('{}={}'.format(k, getattr(self.rc, k)) for k in keys)
        return 'MM: {}'.format(spec)


def get_datastore_path(path):
    datastore_path = Path(path)
    if not datastore_path.is_dir():
        datastore_path = datastore_path.parent
    return datastore_path


def load_info(path):
    datastore_path = get_datastore_path(path)
    with open(str(datastore_path.joinpath('info.json'))) as file:
        info = json.load(file)
    info['datastore_path'] = datastore_path
    return info


module_records_map = {
    'bptt_wgan': GANRecords,
    'bptt_cwgan': ConditionalGANRecords,
    'bptt_moments': MomentMatchingRecords,
    # Legacy GAN modules.  Not well supported, but why not let them be
    # loaded:
    'gan': GANRecords,
    'cgan': GANRecords,
}


def load_records(path):
    datastore_path = get_datastore_path(path)
    info = load_info(datastore_path)
    run_module = guess_run_module(info)
    return module_records_map[run_module].from_info(info, datastore_path)
