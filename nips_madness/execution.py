import json
import os

import lasagne
import numpy
import theano

from . import utils


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


class DataStore(object):

    def __init__(self, directory):
        self.directory = directory
        self.tables = DataTables(directory)

    def path(self, *subpaths):
        newpath = os.path.join(self.directory, *subpaths)
        makedirs_exist_ok(os.path.dirname(newpath))
        return newpath

    def dump_json(self, obj, filename):
        with open(self.path(filename), 'w') as fp:
            json.dump(obj, fp)

    def __repr__(self):
        return '<DataStore: {}>'.format(self.directory)


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

    meta_info = utils.get_meta_info(packages=packages)
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
    run_config = pre_learn(packages=packages, extra_info=extra_info,
                           preprocess=preprocess,
                           **run_config)
    datastore = DataStore(run_config.pop('datastore'))
    with datastore.tables:
        return learn(datastore=datastore, **run_config)
