import numpy as np
import pandas
import pytest

from ..disc_learning import DiscriminatorLog
from ..loader import GANData, parse_tag


def dummy_gandata(gen=None):
    if gen is None:
        gen = np.zeros((5, (1 + 3 * 4)))
        gen[:, 1:] = np.arange(5 * 3 * 4).reshape((5, -1))
        gen[:, 0] = np.arange(5)
    return GANData(**locals())


def test_gen_data():
    data = dummy_gandata()
    np.testing.assert_equal(data.log_J[0].flatten(), np.arange(4))


def test_parse_tag():
    assert parse_tag('DEBUG') == {}
    assert parse_tag('generated_asym_tanh_CE') == dict(
        use_data=False, io_type='asym_tanh', loss='CE',
        layers=[], rate_cost=0,
    )
    assert parse_tag('data_asym_tanh_CE') == dict(
        use_data=True, io_type='asym_tanh', loss='CE',
        layers=[], rate_cost=0,
    )
    assert parse_tag('data_asym_tanh_CE_128_128') == dict(
        use_data=True, io_type='asym_tanh', loss='CE',
        layers=[128, 128], rate_cost=0,
    )
    assert parse_tag('data_asym_tanh_CE_128_1.28') == dict(
        use_data=True, io_type='asym_tanh', loss='CE',
        layers=[128], rate_cost=1.28,
    )


def test_disc_log_init_different_rows():
    learning = pandas.DataFrame(
        np.array([[0] * 3,
                  range(3),
                  range(3),
                  range(3)]).T,
        columns=['gen_step', 'disc_step',
                 'another_column', 'yet_another_column'],
    )
    param_stats = pandas.DataFrame(
        np.array([[0] * 5,
                  range(5),
                  range(5)]).T,
        columns=['gen_step', 'disc_step', 'another_column'],
    )

    class PatchedDiscriminatorLog(DiscriminatorLog):
        def get_info(self):
            return dict(run_config=dict(
                batchsize=1,
                truth_size=100,
            ))

    with pytest.warns(None) as record:
        disc = PatchedDiscriminatorLog(learning, param_stats)
    assert len(record) == 1
    assert len(disc.learning) == len(disc.param_stats) == 3
