import pytest

from . import test_cwgan
from . import test_wgan
from ..dataset import generate_dataset, dataset_provider_choices

smoke_test_configs = [
    {},
    dict(hide_cell_type=True),
    dict(ssn_type='heteroin'),
    dict(ssn_type='deg-heteroin'),
]


@pytest.mark.parametrize('config', smoke_test_configs)
@pytest.mark.parametrize('dataset_provider', dataset_provider_choices)
def test_smoke_generate_dataset_from_wgan(config, dataset_provider,
                                          emit_gan=test_wgan.emit_gan):
    gan, _ = emit_gan(**config)
    generate_kwargs = dict(
        dataset_provider=dataset_provider,
        truth_seed=0,
        truth_size=1,
    )
    if (config.get('ssn_type') in ('heteroin', 'deg-heteroin') and
            dataset_provider == 'ssnode'):
        with pytest.raises(NotImplementedError):
            generate_dataset(gan, **generate_kwargs)
        return
    generate_dataset(gan, **generate_kwargs)


@pytest.mark.parametrize('config', smoke_test_configs)
@pytest.mark.parametrize('dataset_provider', dataset_provider_choices)
def test_smoke_generate_dataset_from_cwgan(config, dataset_provider):
    test_smoke_generate_dataset_from_wgan(config, dataset_provider,
                                          test_cwgan.emit_gan)
