import numpy
import pytest

from ... import ssnode
from ..bptt_gan import DEFAULT_PARAMS, grid_stimulator_inputs
from ..ssn import BandwidthContrastStimulator, EulerSSNModel


def make_ssn(model_config):
    kwds = dict(model_config)
    kwds.pop('bandwidths', None)
    kwds.pop('contrasts', None)
    kwds.pop('sample_sites', None)
    kwds.pop('batchsize', None)
    stimulator, kwds = BandwidthContrastStimulator.consume_kwargs(**kwds)
    model, kwds = EulerSSNModel.consume_kwargs(stimulator, **kwds)
    assert not kwds
    return model


@pytest.mark.parametrize('num_sites', [10])
def test_compare_with_ssnode(num_sites):
    batchsize = 1
    bandwidths = DEFAULT_PARAMS['bandwidths']
    contrasts = DEFAULT_PARAMS['contrasts']
    stimulator_contrasts, stimulator_bandwidths \
        = grid_stimulator_inputs(contrasts, bandwidths, batchsize)
    num_tcdom = stimulator_bandwidths.shape[-1]
    seqlen = 4000  # 3000 was too small
    # seqlen = DEFAULT_PARAMS['seqlen']
    skip_steps = seqlen - 1

    model = make_ssn(dict(
        DEFAULT_PARAMS,
        num_sites=num_sites,
        num_tcdom=num_tcdom,
        seqlen=seqlen,
        skip_steps=skip_steps,
        J=ssnode.DEFAULT_PARAMS['J'],
        D=ssnode.DEFAULT_PARAMS['D'],
        S=ssnode.DEFAULT_PARAMS['S'],
    ))

    # ssnode_fps.shape: (batchsize, num_tcdom, 2N)
    zs, ssnode_fps, info = ssnode.sample_fixed_points(
        batchsize,
        N=num_sites,
        bandwidths=bandwidths,
        contrast=contrasts,
    )

    # time_avg.shape: (batchsize, num_tcdom, 2N)
    time_avg = model.compute_time_avg(
        zs, stimulator_bandwidths, stimulator_contrasts)

    numpy.testing.assert_allclose(time_avg, ssnode_fps, rtol=1e-4, atol=5e-4)
    # Those tolerance settings were too small (even for seqlen=10000):
    # numpy.testing.assert_allclose(time_avg, ssnode_fps, rtol=1e-5, atol=5e-4)
    # numpy.testing.assert_allclose(time_avg, ssnode_fps, rtol=1e-6, atol=5e-4)
    # numpy.testing.assert_allclose(time_avg, ssnode_fps, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('num_sites', [201])
def test_compare_with_ssnode_slowtest(num_sites):
    test_compare_with_ssnode(num_sites)
