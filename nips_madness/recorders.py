from types import SimpleNamespace
import collections
import itertools

import lasagne
import numpy as np

from .networks.ssn import concat_flat


class UpdateResult(SimpleNamespace):
    """
    Result of a generator update.

    Attributes
    ----------
    Dloss : float
        Value of the discriminator loss.  If there are multiple
        discriminator updates per generator update (in WGAN), this is
        the value from the last iteration.
    Gloss : float
        Value of the generator loss.
    Daccuracy : float
        Accuracy of generator.  For WGAN, it's the estimate of
        negative Wasserstein distance.
    rtest : array of shape (NZ, len(inp), 2N)
        Fixed-points of the fake SSN.  This is the second element
        (`Rs`) returned by `.find_fixed_points`.
    true : array
        True tuning curves.  This is the fixed-point "reduced" by
        `.subsample_neurons`.  The shape of this array is
        ``(NZ, NB * len(sample_sites))`` if `track_offset_identity` is
        set to true or otherwise ``(NZ * len(sample_sites), NB)``.
        See `make_WGAN_functions` and  `make_RGAN_functions`.
    model_info : `.FixedPointsInfo` or `.SimpleNamespace`
        An object with attribute `rejections` and `unused` which are
        the sum of the value in the same field of all
        `.FixedPointsInfo` objects returned by `.find_fixed_points`
        calls in `.WGAN_update`.  In case of `.RGAN_update`, it simply
        is `.FixedPointsInfo` since there is only one generator
        update.
    SSsolve_time : float
        Total wall time spent for solving fixed points in update
        function.
    gradient_time : float
        Total wall time spent for calculating gradient and updating
        parameters.
    rate_penalty : float
        Rate penalty evaluated on `rtest`.

    Todo
    ----
    Rename parameters like `.rtest`, `.true` to more descriptive ones.

    """

    dynamics_penalty = np.nan


class BaseRecorder(object):

    def __init__(self, datastore, quiet=True):
        self.datastore = datastore
        self.quiet = quiet

    @property
    def column_names(self):
        return self.dtype.names

    def record(self, *row):
        self._saverow(row)

    @classmethod
    def make(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        self.write_header()
        return self

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore)


class CSVRecorder(BaseRecorder):

    @property
    def filename(self):
        return self.tablename + '.csv'

    def _saverow(self, row):
        assert len(row) == len(self.column_names)
        self.datastore.tables.saverow(self.filename, row, echo=not self.quiet)

    def write_header(self):
        self._saverow(self.column_names)


class HDF5Recorder(BaseRecorder):

    dedicated = False

    def _saverow(self, row):
        typed_row = np.array(tuple(row), dtype=self.dtype)
        self.datastore.h5.tables.saverow(self.tablename,
                                         typed_row,
                                         echo=not self.quiet)

    def write_header(self):
        self.datastore.h5.tables.create_table(self.tablename, self.dtype,
                                              dedicated=self.dedicated)


class LearningRecorder(HDF5Recorder):

    tablename = 'learning'
    dtype = np.dtype([
        ('gen_step', 'uint32'),
        ('Gloss', 'double'),
        ('Dloss', 'double'),
        ('Daccuracy', 'double'),
        ('SSsolve_time', 'double'),
        ('gradient_time', 'double'),
        ('model_convergence', 'uint32'),
        ('model_unused', 'uint32'),
        ('rate_penalty', 'double'),
        ('dynamics_penalty', 'double'),
    ])

    def record(self, gen_step, update_result):
        self._saverow([
            gen_step,
            update_result.Gloss,
            update_result.Dloss,
            update_result.Daccuracy,
            update_result.SSsolve_time,
            update_result.gradient_time,
            update_result.model_info.rejections,
            update_result.model_info.unused,
            update_result.rate_penalty,
            update_result.dynamics_penalty,
        ])

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, quiet=driver.quiet)


class MMLearningRecorder(HDF5Recorder):

    tablename = 'learning'
    dtype = np.dtype([
        ('step', 'uint32'),
        ('loss', 'double'),
        ('rate_penalty', 'double'),
        ('dynamics_penalty', 'double'),
        ('train_time', 'double'),
    ])

    def record(self, gen_step, update_result):
        self._saverow([
            gen_step,
            update_result.loss,
            update_result.rate_penalty,
            update_result.dynamics_penalty,
            update_result.train_time,
        ])

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, quiet=driver.quiet)


class GenMomentsRecorder(HDF5Recorder):

    tablename = 'gen_moments'

    def __init__(self, datastore, num_mom_conds):
        super(GenMomentsRecorder, self).__init__(datastore)
        self.num_mom_conds = num_mom_conds

        self.dtype = np.dtype([
            ('step', 'uint32'),
        ] + [
            ('mean_{}'.format(i), 'double') for i in range(num_mom_conds)
        ] + [
            ('var_{}'.format(i), 'double') for i in range(num_mom_conds)
        ])

    def record(self, gen_step, update_result):
        self._saverow([gen_step] + list(update_result.gen_moments.flat))
    # gen_moments is calculated in
    # [[./networks/moment_matching.py::^def sample_moments]]

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, driver.mmatcher.num_mom_conds)


class DiscLearningRecorder(HDF5Recorder):

    tablename = 'disc_learning'
    dtype = np.dtype([
        ('gen_step', 'uint32'),
        ('disc_step', 'uint32'),
        ('Dloss', 'double'),
        ('Daccuracy', 'double'),
        ('SSsolve_time', 'double'),
        ('gradient_time', 'double'),
        ('model_convergence', 'uint32'),
        ('model_unused', 'uint32'),
    ])


def _genparam_names():
    """
    >>> _genparam_names()                              # doctest: +ELLIPSIS
    ('J_EE', 'J_EI', 'J_IE', 'J_II', 'D_EE', ...)
    """
    def names(prefix):
        J = (prefix + '_{}').format
        return np.array([
            [J('EE'), J('EI')],
            [J('IE'), J('II')],
        ])
    return tuple(concat_flat([names('J'), names('D'), names('S')]))


def gen_param_dtype(names):
    return [
        ('gen_step', 'uint32'),
    ] + [
        (n, 'double') for n in names
    ]


class GenParamRecorder(HDF5Recorder):

    tablename = 'generator'
    dtype = np.dtype(gen_param_dtype(_genparam_names()))

    def __init__(self, datastore, gan):
        self.gan = gan
        super(GenParamRecorder, self).__init__(datastore)

    def record(self, gen_step):
        jj, dd, ss = self.gan.get_gen_param()

        self._saverow([gen_step] + list(np.concatenate([jj, dd, ss]).flat))
        return [jj, dd, ss]

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, driver.gan)


class FlexGenParamRecorder(GenParamRecorder):
    """
    Flexible version of `GenParamRecorder` with ssn_type=heteroin support.
    """

    def __init__(self, *args, **kwargs):
        super(FlexGenParamRecorder, self).__init__(*args, **kwargs)

        self.dtype = np.dtype(gen_param_dtype(
            self.gan.gen.get_flat_param_names()))

    def record(self, gen_step):
        self._saverow([gen_step] + list(self.gan.gen.get_flat_param_values()))
        return self.gan.get_gen_param()  # for compatibility  # TODO: remove!


class DiscParamStatsRecorder(HDF5Recorder):

    tablename = 'disc_param_stats'

    def __init__(self, datastore, discriminator):
        self.discriminator = discriminator
        super(DiscParamStatsRecorder, self).__init__(datastore)

        params = lasagne.layers.get_all_params(discriminator, trainable=True)
        self.dtype = np.dtype([
            ('gen_step', 'uint32'),
            ('disc_step', 'uint32'),
        ] + [
            (name, 'double')
            for name in self.disc_param_unique_names(p.name for p in params)
        ])

    @staticmethod
    def disc_param_unique_names(names):
        counter = collections.Counter()
        for n in names:
            yield '{}.nnorm.{}'.format(n, counter[n])  # Normalized NORM
            counter[n] += 1

    def record(self, gen_step, disc_step):
        nnorms = [
            np.linalg.norm(arr.flatten()) / arr.size
            for arr in lasagne.layers.get_all_param_values(self.discriminator,
                                                           trainable=True)
        ]
        # TODO: implement it using Theano function
        self._saverow([gen_step, disc_step] + nnorms)
        return nnorms

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, driver.gan.discriminator)


class ConditionalTuningCurveStatsRecorder(HDF5Recorder):

    tablename = "tc_stats"
    dedicated = True

    def __init__(self, datastore, num_bandwidths):
        super(ConditionalTuningCurveStatsRecorder, self).__init__(datastore)
        self.num_bandwidths = num_bandwidths

        self.dtype = np.dtype([
            ('gen_step', 'uint32'),
            ('is_fake', 'b'),
            ('contrast', 'double'),
            ('norm_probe', 'double'),
            ('cell_type', 'uint16'),
            ('count', 'uint32'),
        ] + [
            ('mean_{}'.format(i), 'double') for i in range(num_bandwidths)
        ] + [
            ('var_{}'.format(i), 'double') for i in range(num_bandwidths)
        ])

    @staticmethod
    def analyze(tuning_curves, conditions):
        def key(i):
            return tuple(conditions[i])
        indices = sorted(range(len(conditions)), key=key)

        for cond, group in itertools.groupby(indices, key=key):
            tc = tuning_curves[list(group)]
            row = list(cond)     # contrast, norm_probe, cell_type
            row.append(len(tc))  # count
            row.extend(tc.mean(axis=0))
            row.extend(tc.var(axis=0))
            yield row

    def record(self, gen_step, info):
        # `info` is the object returned by train_discriminator
        # See: [[./networks/cwgan.py::train_discriminator]]

        for is_fake, x, c in [(0, info.xd, info.cd),
                              (1, info.xg, info.cg)]:
            for cond_stats in self.analyze(x, c):
                self._saverow([gen_step, is_fake] + list(cond_stats))

    @classmethod
    def from_driver(cls, driver):
        num_bandwidths = len(driver.gan.bandwidths)
        return cls.make(driver.datastore, num_bandwidths)
