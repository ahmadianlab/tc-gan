from pathlib import Path
import json

from ..analyzers.loader import guess_run_module
from ..utils import cached_property
from .datastore_loader import get_datastore
from .run_configs import get_run_config


def cached_record(name):
    def get(self, name):
        df = self.datastore.load(name)
        if 'disc_step' in df.columns:
            disc_step = df.loc[:, 'disc_step']
            df.loc[:, 'epochs'] = self.rc.disc_updates_to_epoch(disc_step + 1)
        elif 'gen_step' in df.columns:
            gen_step = df.loc[:, 'gen_step']
            df.loc[:, 'epochs'] = self.rc.gen_step_to_epoch(gen_step)
        return df
    get.__name__ = name
    return cached_property(get)


class BaseRecords(object):

    def __init__(self, datastore, info, rc):
        self.datastore = datastore
        self.info = info
        self.rc = rc

    learning = cached_record('learning')
    generator = cached_record('generator')

    @classmethod
    def from_info(cls, info, datastore_path):
        rc = get_run_config(info)
        datastore = get_datastore(datastore_path, info)
        return cls(datastore, info, rc)


class GANRecords(BaseRecords):

    disc_param_stats = cached_record('disc_param_stats')
    disc_learning = cached_record('disc_learning')


class ConditionalGANRecords(GANRecords):

    tc_stats = cached_record('tc_stats')


class MomentMatchingRecords(BaseRecords):

    gen_moments = cached_record('gen_moments')


def get_datastore_path(path):
    datastore_path = Path(path)
    if not datastore_path.is_dir():
        datastore_path = datastore_path.parent
    return datastore_path


def load_info(path):
    datastore_path = get_datastore_path(path)
    with open(str(datastore_path.joinpath('info.json'))) as file:
        info = json.load(file)
    info['datastore_path'] = datastore_path
    return info


module_records_map = {
    'bptt_wgan': GANRecords,
    'bptt_cwgan': ConditionalGANRecords,
    'bptt_moments': MomentMatchingRecords,
    # Legacy GAN modules.  Not well supported, but why not let them be
    # loaded:
    'gan': GANRecords,
    'cgan': GANRecords,
}


def load_records(path):
    datastore_path = get_datastore_path(path)
    info = load_info(datastore_path)
    run_module = guess_run_module(info)
    return module_records_map[run_module].from_info(info, datastore_path)
