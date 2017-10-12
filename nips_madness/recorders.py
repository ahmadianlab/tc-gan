from types import SimpleNamespace

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

    @classmethod
    def make(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        self.write_header()
        return self


class LearningRecorder(BaseRecorder):

    filename = "learning.csv"
    column_names = (
        "epoch", "Gloss", "Dloss", "Daccuracy", "SSsolve_time",
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
            for arr in lasagne.layers.get_all_param_values(self.discriminator)
        ]
        # TODO: implement it using Theano function
        self._saverow([gen_step, disc_step] + nnorms)
        return nnorms

    @classmethod
    def from_driver(cls, driver):
        return cls.make(driver.datastore, driver.gan.discriminator)
