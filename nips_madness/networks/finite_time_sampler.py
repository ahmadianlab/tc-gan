import numpy as np

from .. import ssnode
from ..gradient_expressions.utils import sample_sites_from_stim_space
from .bptt_gan import DEFAULT_PARAMS, grid_stimulator_inputs
from .ssn import make_tuning_curve_generator
from .utils import largerrecursionlimit


class FiniteTimeTuningCurveSampler(object):

    @classmethod
    def from_dict(cls, dct):
        default = dict(
            DEFAULT_PARAMS,
            J=ssnode.DEFAULT_PARAMS['J'],
            D=ssnode.DEFAULT_PARAMS['D'],
            S=ssnode.DEFAULT_PARAMS['S'],
            seed=0,
        )

        self, rest = cls.consume_kwargs(**dict(default, **dct))
        assert not rest
        return self

    @classmethod
    def consume_kwargs(cls, bandwidths, contrasts, seed,
                       sample_sites, num_sites, consume_union=True, **kwargs):
        gen, rest = make_tuning_curve_generator(
            kwargs,
            consume_union=consume_union,
            # Stimulator:
            num_tcdom=len(bandwidths) * len(contrasts),
            num_sites=num_sites,
            # Prober:
            probes=sample_sites_from_stim_space(sample_sites, num_sites),
        )
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
            = grid_stimulator_inputs(contrasts, bandwidths, self.batchsize)

    num_neurons = property(lambda self: self.num_sites * 2)
    num_sites = property(lambda self: self.gen.num_sites)
    batchsize = property(lambda self: self.gen.batchsize)

    def forward(self, full_output=False):
        out = self.gen.forward(
            self.rng,
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
        )
        if full_output:
            return out
        return out.prober_tuning_curve

    def compute_trajectories(self):
        trajectories = self.gen.model.compute_trajectories(
            rng=self.rng,
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
        )
        return trajectories

    def timepoints(self):
        dt = self.gen.model.dt
        seqlen = self.gen.model.seqlen
        ts = np.linspace(dt, dt * seqlen, seqlen)
        return ts

    @property
    def dom_points(self):
        for contrast in self.contrasts:
            for bandwidth in self.bandwidths:
                yield dict(contrast=contrast, bandwidth=bandwidth)

    def prepare(self):
        """ Force compile Theano functions. """
        with largerrecursionlimit(self.gen.model.unroll_scan,
                                  self.gen.model.seqlen):
            self.gen.prepare()
