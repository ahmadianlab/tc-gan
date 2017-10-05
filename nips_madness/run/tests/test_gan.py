from unittest.mock import Mock

import numpy as np
import pytest

from .. import gan
from ... import execution
from ... import ssnode


def single_g_step(args):
    gan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
        '--WGAN_n_critic0', '1',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--loss', 'CE'],
    ['--track_offset_identity'],
    ['--sample-sites', '0, 0.5'],
    ['--disc-param-save-on-error'],
    ['--gen-update', 'rmsprop'],
])
def test_smoke_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()


def test_disc_param_save_slowtest(cleancwd, single_g_step=single_g_step):
    single_g_step([
        '--disc-param-save-interval', '1',
        '--datastore', '.',
    ])
    assert cleancwd.join('disc_param', 'last.npz').check()


def test_quit_JDS_threshold_quit():
    datastore = Mock()
    JDS_true = [ssnode.DEFAULT_PARAMS[k] for k in 'JDS']
    log_JDS = [  # from t017/28
        [[-2.725132882048388, -2.4531698490543286],
         [-2.1680251198864506, -3.1575330875403287]],
        [[-0.6751161156746839, -0.3826601625506246],
         [-0.34232003427022, -1.1335422836893538]],
        [[-2.7327504049935936, -4.210179719643937],
         [-1.9547447679855652, -3.5791486928972325]],
    ]
    with pytest.raises(execution.SuccessExit):
        gan.maybe_quit(
            datastore,
            JDS_fake=list(map(np.exp, log_JDS)),
            JDS_true=JDS_true,
            quit_JDS_threshold=0.4,
        )
    datastore.dump_json.assert_called_once()


def test_quit_JDS_threshold_noquit():
    datastore = Mock()
    JDS_true = [ssnode.DEFAULT_PARAMS[k] for k in 'JDS']
    JDS_fake = np.array(JDS_true) + 0.01
    gan.maybe_quit(
        datastore,
        JDS_fake=JDS_fake,
        JDS_true=JDS_true,
        quit_JDS_threshold=0.4,
    )
    datastore.dump_json.assert_not_called()
