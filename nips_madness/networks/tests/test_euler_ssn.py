import numpy
import numpy as np
import pytest

from ... import ssnode
from ...utils import report_allclose_tols
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


def _JDS_for_test():
    # Original SSN parameters:
    J = np.array([[.0957, .0638], [.1197, .0479]])
    D = np.array([[.7660, .5106], [.9575, .3830]])
    S = np.array([[.6667, .2], [1.333, .2]]) / 8

    # More stable parameters:
    D_new = D / 2
    J_new = J + D / 2 - D_new / 2
    return dict(J=J_new, D=D_new, S=S)

JDS = _JDS_for_test()


@pytest.mark.parametrize('num_sites, batchsize', [
    (10, 1),
    (10, 2),
    # (10, 100),  # worked, but slow (~18 sec)
    # (201, 100),  # worked, but very slow
])
def test_compare_with_ssnode(num_sites, batchsize,
                             seqlen=4000, rtol=5e-4, atol=5e-4):
    seed = num_sites * batchsize  # let's co-vary seed as well
    bandwidths = DEFAULT_PARAMS['bandwidths']
    contrasts = DEFAULT_PARAMS['contrasts']
    stimulator_contrasts, stimulator_bandwidths \
        = grid_stimulator_inputs(contrasts, bandwidths, batchsize)
    num_tcdom = stimulator_bandwidths.shape[-1]
    skip_steps = seqlen - 1

    model = make_ssn(dict(
        DEFAULT_PARAMS,
        num_sites=num_sites,
        num_tcdom=num_tcdom,
        seqlen=seqlen,
        skip_steps=skip_steps,
        **JDS
    ))

    # ssnode_fps.shape: (batchsize, num_tcdom, 2N)
    zs, ssnode_fps, info = ssnode.sample_fixed_points(
        batchsize,
        N=num_sites,
        bandwidths=bandwidths,
        contrast=contrasts,
        seed=seed,
        io_type=model.io_type,
        atol=1e-10,
        **JDS
    )

    # time_avg.shape: (batchsize, num_tcdom, 2N)
    time_avg = model.compute_time_avg(
        zs, stimulator_bandwidths, stimulator_contrasts)

    report_allclose_tols(time_avg, ssnode_fps,
                         rtols=[1e-2, 1e-3, 5e-4, 1e-4],
                         atols=[1e-2, 1e-3, 5e-4, 1e-4])

    numpy.testing.assert_allclose(time_avg, ssnode_fps, rtol=rtol, atol=atol)


@pytest.mark.parametrize('num_sites, batchsize', [
    (201, 1),
    # (201, 100),  # worked, but very slow
    (10, 100),
])
def test_compare_with_ssnode_slowtest(num_sites, batchsize):
    test_compare_with_ssnode(num_sites, batchsize,
                             seqlen=10000, rtol=1e-4)
