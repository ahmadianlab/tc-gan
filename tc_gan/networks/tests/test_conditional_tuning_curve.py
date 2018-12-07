import itertools

import numpy as np
import pytest

from ... import ssnode
from ...core import BaseComponent, consume_config
from ...utils import report_allclose_tols
from ..cwgan import RandomChoiceSampler
from .test_cwgan import emit_gan
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
            indices = batch_indices.tc_md[:, :, 0]
            sorted_indices = sorted(set(indices.flat))
            padded_indices = np.zeros(batch_data.num_models, dtype=int)
            padded_indices[:len(sorted_indices)] = sorted_indices  # [1]_
            zs = self.zs[padded_indices]
            ssnode_fps = self.rates[padded_indices]
            model_ids = np.array(  # [1]_
                list(map(sorted_indices.index, indices.flatten())),
                dtype='uint16')
            yield batch_data, zs, ssnode_fps, model_ids

# .. [1] In order to feed zs from true dataset into generator, I need
#    ``num_models >= truth_size`` since `RandomChoiceSampler` does not
#    limit number of sample IDs.  See how sample IDs are squeezed into
#    an array (`padded_indices`) of length `num_models` in
#    `SamplerWrapper.random_minibatches`.


def params_test_compare_with_sample_tuning_curves():
    for (
            (truth_size, probes_per_model),
            norm_probes,
            include_inhibitory_neurons,
    ) in itertools.product(
        [(1, 1),
         (8, 2)],
        [[0], [0, 0.5]],
        [False, True],
    ):

        max_probes = len(norm_probes)
        if include_inhibitory_neurons:
            max_probes *= 2
        if probes_per_model > max_probes:
            continue

        yield truth_size, probes_per_model, norm_probes, \
            include_inhibitory_neurons


@pytest.mark.parametrize(
    'truth_size, probes_per_model, norm_probes'
    ', include_inhibitory_neurons',
    list(params_test_compare_with_sample_tuning_curves()))
def test_compare_with_sample_tuning_curves(
        truth_size, probes_per_model, norm_probes,
        include_inhibitory_neurons,
        seqlen=4000, rtol=5e-4, atol=5e-4):
    gan, rest = emit_gan(
        J0=JDS['J'],
        D0=JDS['D'],
        S0=JDS['S'],
        truth_seed=1,
        truth_size=truth_size,
        num_models=truth_size,   # num_models >= truth_size reqruired [1]_
        probes_per_model=probes_per_model,
        norm_probes=norm_probes,
        include_inhibitory_neurons=include_inhibitory_neurons,
        seqlen=seqlen,
        skip_steps=seqlen - 1,
        include_time_avg=True,
    )
    gen = gan.gen
    sampler, rest = consume_config(
        SamplerWrapper.consume_kwargs,
        rest, gan,
        sampler_class=RandomChoiceSampler,
    )
    dataset = sampler.random_minibatches()

    for _ in range(1):
        batch, zs, ssnode_fps, model_ids = next(dataset)
        assert zs.shape == (gan.num_models, gen.num_neurons, gen.num_neurons)
        gen_kwargs = batch.gen_kwargs
        gen_kwargs['prober_model_ids'] = model_ids  # [1]_
        gen_kwargs['model_rate_penalty_threshold'] = 200
        gen_out = gen.forward(model_zs=zs, **gen_kwargs)
        xd = batch.tuning_curves          # tuning curves from dataset
        xg = gen_out.prober_tuning_curve  # tuning curves from generator

        batch_contrasts, _norm_probes, _cell_types = batch.conditions.T
        prober = gen.prober
        prober_contrasts = prober.contrasts.eval({
            prober.model.stimulator.contrasts:
                gen_kwargs['stimulator_contrasts'],
            prober.model_ids: gen_kwargs['prober_model_ids'],
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
