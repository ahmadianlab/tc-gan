from contextlib import contextmanager
from pathlib import Path
import warnings

import numpy as np
import pandas


def has_csv_header(file):
    pos = file.tell()
    try:
        first = file.read(1024)  # 1024 should be large enough...
        try:
            float(first.split(',', 1)[0])
        except ValueError:
            return True
        else:
            return False
    finally:
        file.seek(pos)


def structured_array_to_dataframe(a):
    return pandas.DataFrame.from_items((k, a[k]) for k in a.dtype.names)
# This is better than from_records(a, columns=a.dtype.names) since it
# maps dtype as well.


class DataStoreLoader1(object):

    def __init__(self, directory):
        self.directory = Path(directory)

    @contextmanager
    def open_if_not(self, file_or_name, *args, **kwargs):
        if hasattr(file_or_name, 'read'):
            yield file_or_name
        else:
            with self.open(file_or_name, *args, **kwargs) as file:
                yield file

    def open(self, fname, *args, **kwargs):
        return open(str(self.directory.joinpath(fname)), *args, **kwargs)

    def read_csv(self, fname, **kwargs):
        with self.open_if_not(fname) as file:
            return pandas.read_csv(file, **kwargs)

    def read_hdf5(self, fname, name, mode='r', **kwargs):
        # TODO: don't re-open fname='store.hdf5' for every call:
        import h5py
        with h5py.File(str(self.directory.joinpath(fname)), 'r') as file:
            return structured_array_to_dataframe(file[name])

    def default_load(self, name):
        csv_name = name + '.csv'
        if self.directory.joinpath(csv_name).exists():
            return self.read_csv(csv_name)

        dedicated_fname = name + '.hdf5'
        if self.directory.joinpath(dedicated_fname).exists():
            return self.read_hdf5(dedicated_fname, name)

        hdf5f_name = 'store.hdf5'
        if self.directory.joinpath(hdf5f_name).exists():
            return self.read_hdf5(hdf5f_name, name)

        filenames = [csv_name, dedicated_fname, hdf5f_name]
        raise RuntimeError('None of {} exists in directory {}'
                           .format(filenames, self.directory))

    def load(self, name):
        try:
            loader = getattr(self, 'load_' + name)
        except AttributeError:
            pass
        else:
            return loader()
        return self.default_load(name)

    def load_truth(self):
        return np.load(self.directory.joinpath('truth.npy'))

    def load_TC_mean(self):
        TC_mean = self.read_csv('TC_mean.csv', header=None)
        TC_mean.columns = pandas.MultiIndex.from_tuples([
            (key, i) for key in ['gen', 'data']
            for i in range(len(TC_mean.columns) // 2)
        ])
        TC_mean['gen_step'] = np.arange(len(TC_mean))
        return TC_mean
    # See: [[../drivers.py::TC_mean.csv]]

    def load_gen_moments(self):
        gen_moments = self.read_csv('gen_moments.csv', header=None)
        gen_moments.columns = pandas.MultiIndex.from_tuples([
            (key, i) for key in ['mean', 'var']
            for i in range(len(gen_moments.columns) // 2)
        ])
        gen_moments['step'] = np.arange(len(gen_moments))
        return gen_moments
    # See: [[../drivers.py::gen_moments.csv]] which saves gen_moments.flat
    # of [[../networks/moment_matching.py::self.gen_moments]] which, in turn,
    # is calculated from [[../networks/moment_matching.py::sample_moments]]

    def load_tc_stats(self):
        raise NotImplementedError


class DataStoreLoader0(DataStoreLoader1):
    """ Loader for data generated by legacy GAN code. """

    def load_generator(self):
        with self.open('generator.csv') as file:
            header = 0 if has_csv_header(file) else None
            generator = self.read_csv(file, header=header)

        if len(generator.columns) == 12:
            warnings.warn('generator.csv only contains 12 columns; '
                          'inserting index columns.')
            generator['gen_step'] = np.arange(len(generator))

        if header is None:
            warnings.warn('generator.csv has no header line; '
                          'assuming the default.')
            generator.columns = ['gen_step',
                                 'J_EE', 'J_EI', 'J_IE', 'J_II',
                                 'D_EE', 'D_EI', 'D_IE', 'D_II',
                                 'S_EE', 'S_EI', 'S_IE', 'S_II']

        return generator

    def load_learning(self):
        with self.open('learning.csv') as file:
            header = 0 if has_csv_header(file) else None
            learning = self.read_csv(file, header=header)

        if header is None:
            warnings.warn('learning.csv has no header line; '
                          'assuming the default.')
            learning.columns = [
                'gen_step', 'Gloss', 'Dloss', 'Daccuracy', 'SSsolve_time',
                'gradient_time', 'model_convergence', 'model_unused',
                'rate_penalty']

        return learning


datastore_loader_map = {
    0: DataStoreLoader0,
    1: DataStoreLoader1,
}


def get_datastore(path, info):
    data_version = info.get('extra_info', {}).get('data_version', 0)
    return datastore_loader_map[data_version](path)
