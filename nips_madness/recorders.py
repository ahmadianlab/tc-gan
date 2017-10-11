from types import SimpleNamespace

import lasagne
import numpy as np

from . import execution


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


def saveheader_disc_param_stats(datastore, discriminator):
    header = ['gen_step', 'disc_step'] + [
        '{}.nnorm'.format(p.name)  # Normalized NORM
        for p in lasagne.layers.get_all_params(discriminator, trainable=True)
    ]
    datastore.tables.saverow('disc_param_stats.csv', header)


def saverow_disc_param_stats(datastore, discriminator, gen_step, disc_step):
    """
    Save normalized norms of `discriminator`.

    This function also checks for finiteness of the parameter and
    raises an error when found.  "Downloading" discriminator parameter
    is (likely) a time-consuming operation so it makes sense to do it
    here.

    """
    nnorms = [
        np.linalg.norm(arr.flatten()) / arr.size
        for arr in lasagne.layers.get_all_param_values(discriminator)
    ]
    row = [gen_step, disc_step] + nnorms
    datastore.tables.saverow('disc_param_stats.csv', row)

    isfinite_nnorms = np.isfinite(nnorms)
    if not isfinite_nnorms.all():
        if not all(np.isfinite(arr).all() for arr in
                   lasagne.layers.get_all_param_values(discriminator)):
            datastore.dump_json(dict(
                reason='disc_param_has_nan',
                isfinite_nnorms=isfinite_nnorms.tolist(),
                good=False,
            ), 'exit.json')
            raise execution.KnownError(
                "Discriminator parameter is not finite.",
                exit_code=3)


class LearningRecorder(object):

    filename = "learning.csv"
    column_names = (
        "epoch", "Gloss", "Dloss", "Daccuracy", "SSsolve_time",
        "gradient_time", "model_convergence", "model_unused",
        "rate_penalty")

    def __init__(self, datastore, quiet):
        self.datastore = datastore
        self.quiet = quiet

    def _saverow(self, row):
        assert len(row) == len(self.column_names)
        self.datastore.tables.saverow(self.filename, row, echo=not self.quiet)

    def write_header(self):
        self._saverow(self.column_names)

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
