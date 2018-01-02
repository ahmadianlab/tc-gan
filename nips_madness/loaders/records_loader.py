from pathlib import Path
import json
import warnings

import numpy as np
import pandas

from ..core import consume_config
from ..networks.ssn import concat_flat
from ..networks.utils import gridify_tc_samples, sampled_tc_axes
from ..utils import cached_property
from .datastore_loader import get_datastore
from .run_configs import get_run_config, guess_run_module, parse_gen_param_name


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

    >>> element_to_array_names(['X_E', 'X_I', 'Y_E', 'Y_I', 'Z'])
    ['X', 'Y', 'Z']

    """
    names = []
    for element in element_names:
        prefix = element.split('_', 1)[0]
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


def prettify_rc_key(key, tex):
    if key in ('J0', 'D0', 'S0', 'V0') and tex:
        return '${}_0$'.format(key[0])
    if key == 'lam' and tex:
        return r'$\lambda$'
    if key.startswith('true_ssn_options.'):
        subkey = key[len('true_ssn_options.'):]
        if subkey in ('J', 'D', 'S', 'V'):
            if tex:
                return r'${}^{{\mathtt{{true}}}}$'.format(subkey)
        return '*{}'.format(subkey)
    if key.startswith('disc_reg_'):  # e.g., disc_reg_l2_decay
        subkey = key[len('disc_reg_'):]
        try:
            prefix, suffix = subkey.split('_')
        except ValueError:
            pass
        else:
            if prefix in ('l1', 'l2') and tex:
                return r'$\ell_{}^{{\mathtt{{{}}}}}$'.format(prefix[1], suffix)
        return subkey
    if key.startswith('gen_'):
        subkey = key[len('gen_'):]
        if subkey in ('dynamics_cost', 'rate_cost'):
            return subkey
    if key.startswith('num_'):
        subkey = key[len('num_'):]
        return '#{}'.format(subkey)
    return key


def prettify_rc_value(value, tex):
    try:
        array = np.array(value)
        assert array.ndim in (1, 2)
        singleton, = set(array.flat)
    except Exception:
        pass
    else:
        return singleton

    return value


def prettify_rc_key_value(key, value, tex=False):
    r"""
    Prettify `key`-`value` pair for `.BaseRecords.pretty_spec`

    >>> prettify_rc_key_value('spam', 'egg')
    'spam=egg'
    >>> prettify_rc_key_value('S0', 1, tex=True)
    '$S_0$=1'
    >>> prettify_rc_key_value('true_ssn_options.V', 0.5, tex=True)
    '$V^{\\mathtt{true}}$=0.5'
    >>> prettify_rc_key_value('disc_reg_l2_decay', 0.1, tex=True)
    '$\\ell_2^{\\mathtt{decay}}$=0.1'
    >>> prettify_rc_key_value('spam', [[0, 0], [0, 0]])
    'spam=0'
    >>> prettify_rc_key_value('spam', [[0, 1], [2, 3]])
    'spam=[[0, 1], [2, 3]]'
    >>> prettify_rc_key_value('gen_rate_cost', 1)
    'rate_cost=1'

    """
    if key == 'moment_weight_type':
        return value + '-mom.'
    return '{}={}'.format(prettify_rc_key(key, tex),
                          prettify_rc_value(value, tex))


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

    generator = cached_record('generator')
    truth = cached_property(lambda self: self.datastore.load('truth'))

    @cached_property
    def learning(self):
        df = self.datastore.load('learning')
        if 'rate_penalty' in df and 'dynamics_penalty' not in df and \
           self.run_module in ('bptt_wgan', 'bptt_cwgan', 'bptt_moments'):
            # In the earlier versions of BPTT-based GANs/MM, the
            # column 'rate_penalty' is used to store
            # 'dynamics_penalty', so that the plotting function can be
            # re-used.  Now that 'rate_penalty' really is rate
            # penalty, the old column has to be renamed:
            df.rename(columns={'rate_penalty': 'dynamics_penalty'},
                      inplace=True)
            warnings.warn(
                'rate_penalty is renamed dynamics_penalty since that was how'
                ' bptt_wgan/bptt_cwgan/bptt_moments used to work.')
        self.insert_epoch_column(df)
        return df

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

    def gen_params_at(self, gen_step):
        """Get generator parameter at `gen_step` as a `dict`."""
        data = self.generator.iloc[gen_step]  # assuming index==gen_step
        params = {}
        for name in self.param_element_names:
            array_name, index = parse_gen_param_name(name)
            if index:
                if array_name not in params:
                    params[array_name] = np.zeros((2,) * len(index))
                params[array_name][index] = data[name]
            else:
                params[name] = data[name]
        return params

    def make_sampler(self, gen_step=-1, **kwargs):
        """
        Get a `.FixedTimeTuningCurveSampler` based on run-time configuration.

        Parameters
        ----------
        gen_step : int
            Generator step from which generator parameter is taken.
            Default to the last step.  See also `gen_params_at`.
        kwargs : dict
            Passed to `.FixedTimeTuningCurveSampler.consume_kwargs`.

        """
        from ..networks.fixed_time_sampler import FixedTimeTuningCurveSampler
        from ..networks.wgan import DEFAULT_PARAMS
        config = {k: DEFAULT_PARAMS[k] for k in [
            # Let's hard-code minimum required keys so that it is easy
            # to notice backward incompatible changes:
            'num_sites', 'smoothness', 'k', 'n',
            'tau_E', 'tau_I', 'dt', 'io_type',
        ]}
        config['seed'] = 0
        config.update(self.rc.dict)
        config.update(self.gen_params_at(gen_step))
        config.update(
            norm_probes=self.rc.norm_probes,
            batchsize=self.rc.batchsize,
        )
        config.update(kwargs)
        sampler, _ = consume_config(
            FixedTimeTuningCurveSampler.consume_kwargs,
            config, **kwargs)
        return sampler

    def truth_grid(self):
        """
        Accessing `.truth` as order-5 array.

        See: `gridify_tc_samples`.
        """
        return gridify_tc_samples(
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

    @property
    def spec_header(self):
        return self.rc.gan_type

    def pretty_spec(self, keys=None, tex=False):
        if keys is None:
            keys = []
        spec = ' '.join(prettify_rc_key_value(k, self.rc[k], tex=tex)
                        for k in keys)
        return '{}: {}'.format(self.spec_header, spec)

    def plot(self, **kwargs):
        from ..analyzers.learning import plot_learning
        return plot_learning(self, **kwargs)


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

    spec_header = 'MM'

    def plot(self, **kwargs):
        from ..analyzers.mm_learning import plot_mm_learning
        return plot_mm_learning(self, **kwargs)


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
