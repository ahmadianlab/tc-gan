import numpy as np
import pytest

from ...core import consume_config
from ..cwgan import ConditionalTuningCurveGenerator
from ..ssn import emit_tuning_curve_generator, ssn_type_choices
from ..wgan import DEFAULT_PARAMS
from .test_euler_ssn import JDS

TEST_PARAMS = dict(
    DEFAULT_PARAMS,
    # Stimulator:
    num_tcdom=10,
    num_sites=50,
    # Model / SSN:
    tau_E=2,
    dt=0.1,
    seqlen=240,
    skip_steps=200,
    # Prober:
    probes=[0],
    **JDS  # Model / SSN
)
del TEST_PARAMS['bandwidths']
del TEST_PARAMS['contrasts']
del TEST_PARAMS['sample_sites']
del TEST_PARAMS['gen']
del TEST_PARAMS['disc']


def emit_tcg_for_test(**kwargs):
    return emit_tuning_curve_generator(**dict(TEST_PARAMS, **kwargs))


def tcg_for_test(config={}, **kwargs):
    tcg, rest = consume_config(emit_tcg_for_test, config, **kwargs)
    assert not rest
    return tcg


def get_param_values(self):
    values = {}
    for p in self.get_all_params():
        values[p.name] = p.get_value()
    return values


@pytest.mark.parametrize('ssn_type, params', [
    ('default', {}),
    # dict(J=0.5),  # unsupported (should I?)
    ('default', dict(J=[[1, 2], [3, 4]])),
    ('default', dict(J=np.array([[1, 2], [3, 4]], dtype=int))),
    ('default', dict(J=np.array([[1, 2], [3, 4]], dtype='float32'))),
    ('heteroin', dict(V=[0.3, 0])),
    ('deg-heteroin', dict(V=0.5)),
])
def test_tcg_set_params(ssn_type, params):
    config = dict(ssn_type=ssn_type)
    tcg = tcg_for_test(config)
    keys = set(params)
    tcg.set_params(params)
    assert keys == set(params)   # set_params must not modify params
    actual = get_param_values(tcg)

    test = {}
    for k in keys:
        test[k] = np.allclose(actual[k], params[k])
        # Manually compare parameters (instead of
        # np.testing.assert_equal) since params[k] may be broadcast to
        # array.

    assert all(test.values())


def test_tcg_set_unknown_params():
    tcg = tcg_for_test()
    with pytest.raises(ValueError) as excinfo:
        tcg.set_params(dict(V=[0.3, 0]))
    assert 'Unknown parameters:' in str(excinfo.value)


flat_param_names = {
    'default': [
        'J_EE', 'J_EI',
        'J_IE', 'J_II',
        'D_EE', 'D_EI',
        'D_IE', 'D_II',
        'S_EE', 'S_EI',
        'S_IE', 'S_II',
    ],
}
flat_param_names['heteroin'] = ['V_E', 'V_I'] + flat_param_names['default']
flat_param_names['deg-heteroin'] = ['V'] + flat_param_names['default']


@pytest.mark.parametrize('ssn_type', ssn_type_choices)
@pytest.mark.parametrize('conditional', [False, True])
def test_tcg_flat_param_names(ssn_type, conditional):
    desired_names = tuple(flat_param_names[ssn_type])
    config = {}
    if conditional:
        config['emit_tcg'] = ConditionalTuningCurveGenerator.consume_kwargs
    tcg = tcg_for_test(config, ssn_type=ssn_type)
    assert tcg.get_flat_param_names() == desired_names
