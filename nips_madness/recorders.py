from types import SimpleNamespace
import itertools

import lasagne
import numpy as np


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


class BaseRecorder(object):

    def __init__(self, datastore, quiet=True):
        self.datastore = datastore
        self.quiet = quiet

    def _saverow(self, row):
        assert len(row) == len(self.column_names)
        self.datastore.tables.saverow(self.filename, row, echo=not self.quiet)

    def write_header(self):
        self._saverow(self.column_names)

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


class HDF5Recorder(BaseRecorder):

    def _saverow(self, row):
        typed_row = np.array(tuple(row), dtype=self.dtype)
        self.datastore.h5.tables.saverow(self.tablename,
                                         typed_row,
                                         echo=not self.quiet)

    def write_header(self):
        self.datastore.h5.tables.create_table(self.tablename, self.dtype)

    @property
    def column_names(self):
        return self.dtype.names


class LearningRecorder(BaseRecorder):

    filename = "learning.csv"
    column_names = (
        "gen_step", "Gloss", "Dloss", "Daccuracy", "SSsolve_time",
        "gradient_time", "model_convergence", "model_unused",
        "rate_penalty")

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
        ])

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, quiet=driver.quiet)


class MMLearningRecorder(BaseRecorder):

    filename = "learning.csv"
    column_names = (
        "step", "loss", "dynamics_penalty", "train_time",
    )

    def record(self, gen_step, update_result):
        self._saverow([
            gen_step,
            update_result.loss,
            update_result.dynamics_penalty,
            update_result.train_time,
        ])

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, quiet=driver.quiet)


class DiscLearningRecorder(BaseRecorder):

    filename = 'disc_learning.csv'
    column_names = (
        'gen_step', 'disc_step', 'Dloss', 'Daccuracy',
        'SSsolve_time', 'gradient_time',
        "model_convergence", "model_unused",
    )


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
    return tuple(np.concatenate([names('J'), names('D'), names('S')]).flat)


class GenParamRecorder(BaseRecorder):

    filename = 'generator.csv'
    column_names = ('gen_step',) + _genparam_names()

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


class DiscParamStatsRecorder(BaseRecorder):

    filename = 'disc_param_stats.csv'

    def __init__(self, datastore, discriminator):
        self.discriminator = discriminator
        super(DiscParamStatsRecorder, self).__init__(datastore)

        params = lasagne.layers.get_all_params(discriminator, trainable=True)
        self.column_names = tuple(['gen_step', 'disc_step'] + [
            '{}.nnorm'.format(p.name)  # Normalized NORM
            for p in params
        ])

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
