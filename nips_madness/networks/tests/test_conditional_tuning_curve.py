import numpy as np
import pytest

from ... import ssnode
from ...core import BaseComponent
from ...utils import report_allclose_tols
from ..cwgan import RandomChoiceSampler
from .test_cwgan import make_gan
from .test_euler_ssn import JDS


class SamplerWrapper(BaseComponent):

    @classmethod
    def consume_kwargs(cls, gan, *args, **kwargs):
        ssn = gan.gen
        J, D, S = gan.get_gen_param()
        kwargs = dict(
            kwargs,
            sample_sites=gan.sample_sites,
            include_inhibitory_neurons=gan.include_inhibitory_neurons,
            num_sites=ssn.num_sites,
            bandwidths=gan.bandwidths,
            contrasts=gan.contrasts,
            J=J,
            D=D,
            S=S,
            num_models=gan.num_models,
            probes_per_model=gan.probes_per_model,
            e_ratio=gan.e_ratio,
            norm_probes=gan.norm_probes,
        )
        return super(SamplerWrapper, cls).consume_kwargs(*args, **kwargs)

    def __init__(
            self, sampler_class,
            num_models, probes_per_model,
            # Sampler:
            e_ratio, norm_probes,
            # Tuning curve solver:
            sample_sites, truth_size, truth_seed, J, D, S,
            bandwidths, contrasts, num_sites,
            include_inhibitory_neurons,
            dt=5e-4,  # as in run.gan
            max_iter=100000,
            io_type='asym_power',
            rate_stop_at=200,
            seed=0,
            ):
        data, samples = ssnode.sample_tuning_curves(
            sample_sites=sample_sites,
            NZ=truth_size,
            seed=truth_seed,
            J=J, D=D, S=S,
            bandwidths=bandwidths,
            contrast=contrasts,
            N=num_sites,
            track_offset_identity=True,
            include_inhibitory_neurons=include_inhibitory_neurons,
            dt=dt,
            io_type=io_type,
            rate_stop_at=rate_stop_at,
        )
        self.data = data.T
        self.zs, self.rates, self.fpinfo = samples

        self.num_models = num_models
        self.probes_per_model = probes_per_model

        indices = np.arange(len(self.data), dtype='uint16')
        indices = np.broadcast_to(indices.reshape((-1, 1)), self.data.shape)
        self.data_sampler = sampler_class.from_grid_data(
            self.data,
            bandwidths=bandwidths,
            contrasts=contrasts,
            norm_probes=norm_probes,
            include_inhibitory_neurons=include_inhibitory_neurons,
            e_ratio=e_ratio,
        )
        self.indices_sampler = sampler_class.from_grid_data(
            indices,
            bandwidths=bandwidths,
            contrasts=contrasts,
            norm_probes=norm_probes,
            include_inhibitory_neurons=include_inhibitory_neurons,
            e_ratio=e_ratio,
        )

    def random_minibatches(self):
        kwargs = dict(
            num_models=self.num_models,
            probes_per_model=self.probes_per_model,
        )
        for batch_data, batch_indices in zip(
                self.data_sampler.random_minibatches(**kwargs),
                self.indices_sampler.random_minibatches(**kwargs)):
            np.testing.assert_equal(self.data_sampler.rng.get_state(),
                                    self.indices_sampler.rng.get_state())
            indices = batch_indices.tc_md[:, 0, 0]
            zs = self.zs[indices]
            ssnode_fps = self.rates[indices]
            yield batch_data, zs, ssnode_fps


@pytest.mark.parametrize('truth_size, num_models, probes_per_model', [
    (1, 1, 1),
    (8, 4, 2),
])
@pytest.mark.parametrize('include_inhibitory_neurons', [False, True])
def test_compare_with_sample_tuning_curves(
        truth_size, num_models, probes_per_model,
        include_inhibitory_neurons,
        seqlen=4000, rtol=5e-4, atol=5e-4):
    gan, rest = make_gan(
        J0=JDS['J'],
        D0=JDS['D'],
        S0=JDS['S'],
        truth_size=1,
        truth_seed=1,
        probes_per_model=1,
        include_inhibitory_neurons=include_inhibitory_neurons,
        seqlen=seqlen,
        skip_steps=seqlen - 1,
        include_time_avg=True,
    )
    gen = gan.gen
    sampler, rest = SamplerWrapper.consume_config(
        rest, gan, RandomChoiceSampler,
    )
    dataset = sampler.random_minibatches()

    for _ in range(1):
        batch, zs, ssnode_fps = next(dataset)
        assert zs.shape == (gan.num_models, gen.num_neurons, gen.num_neurons)
        gen_out = gen.forward(model_zs=zs, **batch.gen_kwargs)
        xd = batch.tuning_curves          # tuning curves from dataset
        xg = gen_out.prober_tuning_curve  # tuning curves from generator

        batch_contrasts, _norm_probes, _cell_types = batch.conditions.T
        prober = gen.prober
        gen_kwargs = batch.gen_kwargs
        prober_contrasts = prober.contrasts.eval({
            prober.model.stimulator.contrasts:
                gen_kwargs['stimulator_contrasts'],
            prober.model_ids: batch.model_ids,
        })
        np.testing.assert_equal(prober_contrasts, batch_contrasts)

        time_avg = gen_out.model_time_avg
        print('allclose(time_avg, ssnode_fps):')
        report_allclose_tols(time_avg, ssnode_fps,
                             rtols=[1e-2, 1e-3, 5e-4, 1e-4],
                             atols=[1e-2, 1e-3, 5e-4, 1e-4])
        np.testing.assert_allclose(time_avg, ssnode_fps,
                                   rtol=rtol, atol=atol)

        print()
        print('allclose(xd, xd):  # generated vs true tuning curves')
        report_allclose_tols(xg, xd,
                             rtols=[1e-2, 1e-3, 5e-4, 1e-4],
                             atols=[1e-2, 1e-3, 5e-4, 1e-4])

        np.testing.assert_allclose(xg, xd, rtol=rtol, atol=atol)
