import collections

import lasagne
import numpy as np

from . import execution
from . import lasagne_param_file
from . import ssnode
from .recorders import LearningRecorder, GenParamRecorder, \
    DiscLearningRecorder, DiscParamStatsRecorder, MMLearningRecorder, \
    ConditionalTuningCurveStatsRecorder, UpdateResult


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
        self.disc_loss_limiter = disc_loss_limiter(self)

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
        self.disc_loss_limiter(Dloss)

    def post_update(self, gen_step, update_result):
        self.learning_recorder.record(gen_step, update_result)
        jj, dd, ss = self.generator_recorder.record(gen_step)

        if self.disc_param_save_interval > 0 \
                and gen_step % self.disc_param_save_interval == 0:
            lasagne_param_file.dump(
                self.gan.discriminator,
                self.datastore.path('disc_param',
                                    self.disc_param_template.format(gen_step)))

        self.datastore.tables.flush_all()

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


def disc_loss_limiter(driver):
    if driver.gan.loss_type == 'WD':
        return WGANDiscLossLimiter.from_driver(driver)
    else:
        return lambda *_, **__: None


class WGANDiscLossLimiter(object):

    def __init__(self, datastore, prob_limit=0.6,
                 wild_disc_loss=10000, hist_length=50):
        self.datastore = datastore
        self.prob_limit = prob_limit
        self.wild_disc_loss = wild_disc_loss
        self.hist_length = hist_length
        self.dloss_hist = collections.deque(maxlen=hist_length)

    def prob_exceed(self):
        return np.mean(abs(np.asarray(self.dloss_hist) > self.wild_disc_loss))

    def should_abort(self, dloss):
        self.dloss_hist.append(dloss)
        return (len(self.dloss_hist) == self.hist_length and
                self.prob_exceed() > self.prob_limit)

    def __call__(self, dloss):
        if self.should_abort(dloss):
            self.datastore.dump_json(dict(
                reason='wild_disc_loss',
                good=False,
            ), 'exit.json')
            raise execution.KnownError(
                "Too many wild discriminator losses.",
                exit_code=4)

    @classmethod
    def from_driver(cls, driver):
        return cls(driver.datastore)


class BPTTWGANDriver(GANDriver):
    # TODO: don't rely on GANDriver

    def run(self, gan):
        learning_it = gan.learning()

        @self.iterate
        def update_func(k):
            # This callback function "connects" gan.learning and drive.iterate.
            while True:
                info = next(learning_it)
                if info.is_discriminator:
                    self.post_disc_update(
                        info.gen_step,
                        info.disc_step,
                        info.disc_loss,
                        info.accuracy,
                        info.gen_time,
                        info.disc_time,
                        ssnode.null_FixedPointsInfo,
                    )
                    disc_info = info
                else:
                    assert info.gen_step == k
                    # Save fake and tuning curves averaged over Zs:
                    data_mean = disc_info.xd.mean(axis=0).tolist()
                    gen_mean = disc_info.xg.mean(axis=0).tolist()
                    self.datastore.tables.saverow('TC_mean.csv',
                                                  gen_mean + data_mean)

                    # If not info.is_discriminator, then the generator
                    # step was just taken.  Let's return a result that
                    # GANDriver understands.
                    return UpdateResult(
                        Gloss=info.gen_loss,
                        Dloss=disc_info.disc_loss,
                        Daccuracy=disc_info.accuracy,
                        SSsolve_time=info.gen_time,
                        gradient_time=info.disc_time,
                        model_info=ssnode.null_FixedPointsInfo,
                        rate_penalty=disc_info.dynamics_penalty,
                        # For BPTTcWGANDriver:
                        info=info,
                        disc_info=disc_info,
                    )
                # See: [[./recorders.py::def record.*update_result]]


class BPTTcWGANDriver(BPTTWGANDriver):

    def post_update(self, gen_step, update_result):
        self.tuning_curve_recorder.record(gen_step, update_result.disc_info)
        super(BPTTcWGANDriver, self).post_update(gen_step, update_result)

    def pre_loop(self):
        super(BPTTcWGANDriver, self).pre_loop()
        self.tuning_curve_recorder \
            = ConditionalTuningCurveStatsRecorder.from_driver(self)


class MomentMatchingDriver(object):

    # TODO: refactor out common code with GANDriver

    def __init__(self, mmatcher, datastore, iterations, quiet,
                 quit_JDS_threshold=-1):
        self.mmatcher = mmatcher
        self.datastore = datastore
        self.iterations = iterations
        self.quiet = quiet
        self.quit_JDS_threshold = quit_JDS_threshold

    # For compatibility with GenParamRecorder:
    gan = property(lambda self: self.mmatcher)

    def pre_loop(self):
        self.learning_recorder = MMLearningRecorder.from_driver(self)
        self.generator_recorder = GenParamRecorder.from_driver(self)

    def post_update(self, gen_step, update_result):
        self.learning_recorder.record(gen_step, update_result)

        self.datastore.tables.saverow(
            'gen_moments.csv',
            list(update_result.gen_moments.flat))

        jj, dd, ss = self.generator_recorder.record(gen_step)

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

        .. function:: update_func(gen_step : int) -> MMUpdateResult

           It must accept a positional argument which is the generator
           update step.  It then must return an instance of
           `.MMUpdateResult` with all mandatory attributes mentioned in
           the document of `.MMUpdateResult`.

        """
        self.pre_loop()
        for gen_step in range(self.iterations):
            self.post_update(gen_step, update_func(gen_step))
        self.post_loop()

    def run(self, learner):
        learning_it = learner.learning()

        @self.iterate
        def update_func(k):
            info = next(learning_it)
            assert info.step == k
            return info
