from getpass import getuser
from logging import getLogger
from socket import gethostname
import json
import os
import subprocess
import sys

import lasagne
import numpy
import theano

from . import utils

logger = getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class KnownError(Exception):
    """Exception with exit code."""

    def __init__(self, message, exit_code=1):
        self.exit_code = exit_code
        super(KnownError, self).__init__(message)


class SuccessExit(KnownError):
    """Exception for successful exit (code=0)."""

    def __init__(self, message):
        super(SuccessExit, self).__init__(message, exit_code=0)


default_tracking_packages = [
    theano, lasagne, numpy,
]
"""
Packages whose versions are recorded.
"""


def makedirs_exist_ok(name):
    try:
        os.makedirs(name)
    except OSError as err:
        if err.errno != 17:
            raise


def get_meta_info(packages=[]):
    return dict(
        repository=dict(
            revision=git_revision(),
            is_clean=git_is_clean(),
        ),
        python=sys.executable,
        packages={p.__name__: p.__version__ for p in packages},
        argv=sys.argv,
        environ=relevant_environ(),
        theano=utils.theano_info(),
        pid=os.getpid(),
        hostname=gethostname(),
        username=getuser(),
    )


def git_revision():
    return git_output(['git', 'rev-parse', 'HEAD']).rstrip()


def git_is_clean():
    return git_output(['git', 'status', '--short',
                       '--untracked-files=no']).strip() == ''


def relevant_environ(_environ=os.environ):
    """relevant_environ() -> dict
    Extract relevant environment variables and return as a `dict`.
    """
    def subenv(prefix):
        return {k: _environ[k] for k in _environ if k.startswith(prefix)}

    environ = {k: _environ[k] for k in [
        'PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH',
        'HOST', 'HOSTNAME', 'USER', 'USERNAME',
    ] if k in _environ}
    environ.update(subenv('SLURM'))
    environ.update(subenv('PBS'))
    environ.update(subenv('OMP'))
    environ.update(subenv('MKL'))
    environ.update(subenv('THEANO'))
    environ.update(subenv('GPU'))  # especially, GPU_DEVICE_ORDINAL
    return environ


def git_output(args):
    return subprocess.check_output(
        args,
        cwd=PROJECT_ROOT,
        universal_newlines=True)


class DataTables(object):

    def __init__(self, directory):
        self.directory = directory
        self._files = {}

    def _open(self, name):
        """Thin wrapper of `open`, for dependency injection in testing."""
        return open(os.path.join(self.directory, name), 'w')

    def _get_file(self, name):
        if name not in self._files:
            self._files[name] = self._open(name)

        return self._files[name]

    def saverow(self, name, row, echo=False, flush=False):
        if isinstance(row, (list, tuple)):
            row = ','.join(map(str, row))

        file = self._get_file(name)
        file.write(row)
        file.write('\n')
        if flush:
            file.flush()

        if echo:
            print(row)

    def flush_all(self):
        for file in self._files.values():
            file.flush()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for name, file in self._files.items():
            try:
                file.close()
            except Exception as err:
                print('Error while closing', name)
                print(err)
                print('ignoring...')


class HDF5Tables(object):

    shared_filename = 'store.hdf5'

    def __init__(self, h5):
        self.h5 = h5
        self._datasets = {}

    def _create_dataset(self, name, dtype, dedicated=False):
        if dedicated:
            file = self.h5.open(name + '.hdf5')
        else:
            file = self.h5.open(self.shared_filename)
        dataset = file.create_dataset(name, (0,), maxshape=(None,),
                                      dtype=dtype)
        return dataset, file

    def _get_dataset(self, name, *args, **kwds):
        if name not in self._datasets:
            self._datasets[name] = self._create_dataset(name, *args, **kwds)
        return self._datasets[name]

    def create_table(self, name, dtype, dedicated=False):
        assert name not in self._datasets
        self._get_dataset(name, dtype, dedicated)

    def saverow(self, name, row, echo=False, flush=False):
        dataset, file = self._get_dataset(name, row.dtype)
        dataset.resize((len(dataset) + 1,))
        dataset[-1] = row

        if flush:
            file.flush()
        if echo:
            print(*row.tolist(), sep=',')


class HDF5Store(object):

    def __init__(self, datastore):
        self.datastore = datastore
        self.tables = HDF5Tables(self)
        self._files = {}

    def _open(self, filename):
        import h5py
        path = os.path.join(self.datastore.directory, filename)
        file = self.datastore.enter_context(h5py.File(path))
        return file

    def open(self, filename):
        if filename not in self._files:
            self._files[filename] = self._open(filename)
        return self._files[filename]

    def flush_all(self):
        for file in self._files.values():
            file.flush()


class DataStore(object):

    def __init__(self, directory):
        self.directory = directory
        self.tables = DataTables(directory)
        self.h5 = HDF5Store(self)
        self.exit_hooks = []

    def path(self, *subpaths):
        newpath = os.path.join(self.directory, *subpaths)
        makedirs_exist_ok(os.path.dirname(newpath))
        return newpath

    def dump_json(self, obj, filename):
        with open(self.path(filename), 'w') as fp:
            json.dump(obj, fp)

    def save_exit_reason(self, reason, good, **kwargs):
        logger.info('Recording reason=%s (%s) in exit.json',
                    reason, 'good' if good else 'bad')
        self.dump_json(dict(reason=reason, good=good, **kwargs), 'exit.json')
        logger.info('exit.json is created.')

    def flush_all(self):
        self.tables.flush_all()
        self.h5.flush_all()

    def __repr__(self):
        return '<DataStore: {}>'.format(self.directory)

    def enter_context(self, context_manager):
        ret = context_manager.__enter__()
        self.exit_hooks.append(context_manager.__exit__)
        return ret

    def __enter__(self):
        self.enter_context(self.tables)
        return self

    def __exit__(self, *exc):
        run_exit_hooks(self.exit_hooks, exc)


def run_exit_hooks(exit_hooks, exc=(None, None, None)):
    if not exit_hooks:
        return
    try:
        exit_hooks[0](*exc)
    finally:
        run_exit_hooks(exit_hooks[1:], exc)
# MAYBE: Do what contextlib.nested was doing.  More complicated.
# See: https://github.com/python/cpython/blob/2.7/Lib/contextlib.py#L89


def format_datastore(datastore_template, run_config):
    """
    Format datastore path based on `run_config` (aka "tagging").

    It is almost equivalent to ``datastore_template.format(**run_config)``
    except that it introduces a new keyword ``layers_str`` based on the
    ``layers`` (list) keyword in `run_config`.

    >>> format_datastore(
    ...     'alpha={alpha}_L={layers_str}',
    ...     dict(alpha=10, layers=[128, 64]))
    'alpha=10_L=128_64'

    """
    return datastore_template.format(
        layers_str='_'.join(map(str, run_config.get('layers', []))),
        **run_config)


def add_base_learning_options(parser):
    """
    Add basic options for controlling learning execution.
    """

    # Datastore related options:
    parser.add_argument(
        '--datastore',
        help='''Directory for output files to be stored.  It is
        created if it does not exist.  If it already exists and
        contains files, they may be overwritten!  If this option is
        not given, directory path is generated according to
        --datastore-template.''')
    parser.add_argument(
        '--datastore-template',
        default='logfiles/{IO_type}_{loss}_{layers_str}_{rate_cost}',
        help='''Python string template to be used for generating
        datastore directory. (default: %(default)s)''')
    parser.add_argument(
        '--debug', dest='datastore_template',
        action='store_const', const='logfiles/debug',
        help='A shorthand for --datastore-template=logfiles/debug.')

    parser.add_argument(
        '--load-config',
        help='''Load configuration (hyper parameters) from a
        JSON/YAML/TOML/Pickle file if given.  Note that configurations
        are overwritten by the ones in the given file if they are
        given by both in command line and the file.''')


def pre_learn(
        packages,
        datastore, datastore_template,
        load_config,
        extra_info={}, preprocess=None,
        **run_config):
    if load_config:
        run_config.update(utils.load_any_file(load_config))

    if preprocess:
        preprocess(run_config)

    if not datastore:
        datastore = format_datastore(datastore_template, run_config)

    makedirs_exist_ok(datastore)

    meta_info = get_meta_info(packages=packages)
    with open(os.path.join(datastore, 'info.json'), 'w') as fp:
        json.dump(dict(
            run_config=run_config,
            extra_info=extra_info,
            meta_info=meta_info,
        ), fp)

    run_config['datastore'] = datastore
    return run_config


def do_learning(learn, run_config, extra_info={}, preprocess=None,
                packages=default_tracking_packages):
    """
    Execute `learn` with `run_config` after pre-processing.

    It is more-or-less equivalent to ``learn(**run_config)`` except
    that keys `datastore_template` and `load_config` are removed and
    `datastore` is an instance of `DataStore` object.

    """
    logger.info('PID: %d', os.getpid())
    run_config = pre_learn(packages=packages, extra_info=extra_info,
                           preprocess=preprocess,
                           **run_config)
    with DataStore(run_config.pop('datastore')) as datastore:
        logger.info('Output directory: %s', datastore.directory)
        return learn(datastore=datastore, **run_config)
