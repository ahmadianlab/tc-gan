import lasagne
import numpy as np

from . import execution
from . import lasagne_param_file
from . import ssnode
from .recorders import LearningRecorder, GenParamRecorder, \
    DiscLearningRecorder, DiscParamStatsRecorder


def net_isfinite(layer):
    return all(np.isfinite(arr).all() for arr in
               lasagne.layers.get_all_param_values(layer))


class GANDriver(object):

    """
    GAN learning driver.

    It does all algorithm-independent ugly stuff such as:

    * Calculate learning statistics.
    * Save learning statistics and parameters at the right point.
    * Make sure the parameters are finite; otherwise save information
      as much as possible and then abort.

    The main interfaces are `iterate` and `post_disc_update`.

    """

    def __init__(self, gan, datastore, **kwargs):
        self.gan = gan
        self.datastore = datastore
        self.__dict__.update(kwargs)

    def pre_loop(self):
        self.learning_recorder = LearningRecorder.from_driver(self)
        self.generator_recorder = GenParamRecorder.from_driver(self)
        self.discparamstats_recorder = DiscParamStatsRecorder.from_driver(self)
        self.disclearning_recorder = DiscLearningRecorder.from_driver(self)
        self.rejection_limiter = SSNRejectionLimiter.from_driver(self)

    def post_disc_update(self, gen_step, disc_step, Dloss, Daccuracy,
                         SSsolve_time, gradient_time, model_info):
        """
        Method to be called after a discriminator update.

        Parameters
        ----------
        gen_step : int
            See `.iterate`
        disc_step : int
            If there are multiple discriminator updates (WGAN), the
            loop index must be given as `disc_step` argument.
            Otherwise, pass 0.
        Dloss : float
        Daccuracy : float
        SSsolve_time : float
        gradient_time : float
        model_info : `.FixedPointsInfo`
            See the attributes of `.UpdateResult` with the same name.

        """
        self.disclearning_recorder.record(
            gen_step, disc_step, Dloss, Daccuracy,
            SSsolve_time, gradient_time,
            model_info.rejections, model_info.unused,
        )

        nnorms = self.discparamstats_recorder.record(gen_step, disc_step)
        check_disc_param(self.datastore, self.gan.discriminator, nnorms)

        self.rejection_limiter(model_info.rejections)

    def post_update(self, gen_step, update_result):
        self.learning_recorder.record(gen_step, update_result)
        jj, dd, ss = self.generator_recorder.record(gen_step)

        if self.disc_param_save_interval > 0 \
                and gen_step % self.disc_param_save_interval == 0:
            lasagne_param_file.dump(
                self.gan.discriminator,
                self.datastore.path('disc_param',
                                    self.disc_param_template.format(gen_step)))

        maybe_quit(
            self.datastore,
            JDS_fake=list(map(np.exp, [jj, dd, ss])),
            JDS_true=list(map(ssnode.DEFAULT_PARAMS.get, 'JDS')),
            quit_JDS_threshold=self.quit_JDS_threshold,
        )

    def post_loop(self):
        self.datastore.dump_json(dict(
            reason='end_of_iteration',
            good=True,
        ), 'exit.json')

    def iterate(self, update_func):
        """
        Iteratively call `update_func` for `.iterations` times.

        A callable `update_func` must have the following signature:

        .. function:: update_func(gen_step : int) -> UpdateResult

           It must accept a positional argument which is the generator
           update step.  It then must return an instance of
           `.UpdateResult` with all mandatory attributes mentioned in
           the document of `.UpdateResult`.

        """
        if self.disc_param_save_on_error:
            update_func = lasagne_param_file.wrap_with_save_on_error(
                self.gan.discriminator,
                self.datastore.path('disc_param', 'pre_error.npz'),
                self.datastore.path('disc_param', 'post_error.npz'),
            )(update_func)

        self.pre_loop()
        for gen_step in range(self.iterations):
            self.post_update(gen_step, update_func(gen_step))
        self.post_loop()


def maybe_quit(datastore, JDS_fake, JDS_true, quit_JDS_threshold):
    JDS_fake = np.concatenate(JDS_fake).flatten()
    JDS_true = np.concatenate(JDS_true).flatten()
    JDS_distance = np.linalg.norm(JDS_fake - JDS_true)

    if quit_JDS_threshold > 0 and JDS_distance >= quit_JDS_threshold:
        datastore.dump_json(dict(
            reason='JDS_distance',
            JDS_distance=JDS_distance,
            good=False,
        ), 'exit.json')
        raise execution.KnownError(
            'Exit simulation since (J, D, S)-distance (= {})'
            ' to the true parameter exceed threshold (= {}).'
            .format(JDS_distance, quit_JDS_threshold),
            exit_code=4)


def check_disc_param(datastore, discriminator, nnorms):
    isfinite_nnorms = np.isfinite(nnorms)
    if not isfinite_nnorms.all() and not net_isfinite(discriminator):
        datastore.dump_json(dict(
            reason='disc_param_has_nan',
            isfinite_nnorms=isfinite_nnorms.tolist(),
            good=False,
        ), 'exit.json')
        raise execution.KnownError(
            "Discriminator parameter is not finite.",
            exit_code=3)


class SSNRejectionLimiter(object):
    """
    Watch out SSN rejection rate and terminate the driver if it is too large.
    """

    def __init__(self, datastore, n_samples,
                 rejection_limit=0.6,
                 max_consecutive_exceedings=5):
        self.datastore = datastore

        self.n_samples = n_samples
        """ Minibatch size; aka NZ """

        self.rejection_limit = rejection_limit
        """ Rate #rejections/#total over which model considered "bad". """

        self.max_consecutive_exceedings = max_consecutive_exceedings
        """ Maximum number of consecutive "bad" models. """

        self._exceedings = 0

    def should_abort(self, rejections):
        if rejections / (rejections + self.n_samples) > self.rejection_limit:
            self._exceedings += 1
        else:
            self._exceedings = 0

        return self._exceedings > self.max_consecutive_exceedings

    def __call__(self, rejections):
        if self.should_abort(rejections):
            self.datastore.dump_json(dict(
                reason='too_many_rejections',
                good=False,
            ), 'exit.json')
            raise execution.KnownError(
                "Too many rejections in fixed-point finder.",
                exit_code=4)

    @classmethod
    def from_driver(cls, driver):
        return cls(driver.datastore, n_samples=driver.gan.NZ)
