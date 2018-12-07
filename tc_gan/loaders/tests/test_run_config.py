from ..run_configs import BaseRunConfig


def test_dotted_key():
    rc = BaseRunConfig(dict(alpha=dict(beta=dict(gamma=1))))
    assert rc['alpha.beta.gamma'] == 1
