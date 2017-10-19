import numpy as np

from .. import ssnode
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..utils import cartesian_product
from .bptt_gan import TuningCurveGenerator


class FiniteTimeTuningCurveSampler(object):

    @classmethod
    def from_dict(cls, dct):
        default = dict(
            bandwidths=ssnode.DEFAULT_PARAMS['bandwidths'],
            contrasts=ssnode.DEFAULT_PARAMS['contrast'],
            smoothness=ssnode.DEFAULT_PARAMS['smoothness'],
            sample_sites=[0],
            seed=0,
            # Stimulator:
            num_sites=ssnode.DEFAULT_PARAMS['N'],
            # Model / SSN:
            J=ssnode.DEFAULT_PARAMS['J'],
            D=ssnode.DEFAULT_PARAMS['D'],
            S=ssnode.DEFAULT_PARAMS['S'],
            k=ssnode.DEFAULT_PARAMS['k'],
            n=ssnode.DEFAULT_PARAMS['n'],
            tau_E=10,
            tau_I=1,
            dt=0.1,
            num_steps=1200,
            batchsize=1,
            skip_steps=1000,
        )

        self, rest = cls.consume_kwargs(**dict(default, **dct))
        assert not rest
        return self

    @classmethod
    def consume_kwargs(cls, bandwidths, contrasts, seed,
                       sample_sites, num_sites, **kwargs):
        gen, rest = TuningCurveGenerator.consume_kwargs(
            # Stimulator:
            num_tcdom=len(bandwidths) * len(contrasts),
            num_sites=num_sites,
            # Prober:
            probes=sample_sites_from_stim_space(sample_sites, num_sites),
            **kwargs)
        return cls(gen, bandwidths, contrasts, seed), rest

    def __init__(self, gen, bandwidths, contrasts, seed):

        bandwidths = np.asarray(bandwidths)
        contrasts = np.asarray(contrasts)
        assert bandwidths.ndim == 1
        assert contrasts.ndim == 1

        self.gen = gen
        self.bandwidths = bandwidths
        self.contrasts = contrasts
        self.rng = np.random.RandomState(seed)

        self.stimulator_contrasts, self.stimulator_bandwidths \
            = cartesian_product(contrasts, bandwidths)

    num_sites = property(lambda self: self.gen.num_sites)
    batchsize = property(lambda self: self.gen.batchsize)

    def forward(self, full_output=False):
        out = self.gen.forward(
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
            model_zs=self.rng.rand(self.batchsize,
                                   self.num_sites,
                                   self.num_sites),
        )
        if full_output:
            return out
        return out.prober_tuning_curve

    def compute_trajectories(self):
        zmat = self.rng.rand(2 * self.num_sites, 2 * self.num_sites)
        trajectories = self.gen.model.compute_trajectories(
            zmat,
            self.stimulator_bandwidths,
            self.stimulator_contrasts,
        )
        return trajectories

    def timepoints(self):
        dt = self.gen.model.dt
        num_steps = self.gen.model.num_steps
        ts = np.linspace(dt, dt * num_steps, num_steps)
        return ts

    @property
    def dom_points(self):
        for contrast in self.contrasts:
            for bandwidth in self.bandwidths:
                yield dict(contrast=contrast, bandwidth=bandwidth)
