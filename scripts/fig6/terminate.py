import pathlib

import pandas

from tc_gan.analyzers.param_table import merge_param_table
from tc_gan.analyzers.termination import simulate_terminations, \
    simulate_terminations_snr
from tc_gan.execution import get_meta_info
from tc_gan.loaders import load_records
from tc_gan.utils import make_progressbar


class TerminationAnalyzer(object):

    def __init__(self, results_directory='results', quiet=False):
        self.results_directory = results_directory
        self.quiet = quiet

        basepath = pathlib.Path(__file__).resolve().parent
        paths_cwgan = basepath.glob('gan/*/store.hdf5')
        paths_moments = basepath.glob('mm/*/store.hdf5')
        self.records_cwgan = list(map(load_records, paths_cwgan))
        self.records_moments = list(map(load_records, paths_moments))

        # They should be non-empty:
        assert self.records_cwgan
        assert self.records_moments

    def log(self, *args):
        if not self.quiet:
            print(*args)

    def load_records(self):
        self.log('Loading records...')
        bar = make_progressbar(self.quiet)
        for rec in bar(self.records_cwgan + self.records_moments):
            rec.generator

    def simulate_terminations(self):
        self.log('Simulating terminations (x4)...')
        progress = not self.quiet
        self.terms_cwgan = pandas.concat([
            simulate_terminations(self.records_cwgan, progress=progress),
            simulate_terminations_snr(self.records_cwgan, progress=progress),
        ], ignore_index=True)
        self.terms_moments = pandas.concat([
            simulate_terminations(self.records_moments, progress=progress),
            simulate_terminations_snr(self.records_moments, progress=progress),
        ], ignore_index=True)

        self.terms_cwgan['method'] = 'GAN'
        self.terms_moments['method'] = 'MM'
        self.terms_cwgan = merge_param_table(self.records_cwgan,
                                             self.terms_cwgan)
        self.terms_moments = merge_param_table(self.records_moments,
                                               self.terms_moments)
        if 'disc_layers' in self.terms_cwgan:
            self.terms_cwgan['disc_layer_width'] = \
                self.terms_cwgan['disc_layers'].map(lambda x: x[0])
            self.terms_cwgan['disc_layer_depth'] = \
                self.terms_cwgan['disc_layers'].map(len)
            del self.terms_cwgan['disc_layers']

        self.terms = pandas.concat(
            [self.terms_cwgan, self.terms_moments],
            ignore_index=True)
        self.terms.index.name = 'term_id'

    def make_param_table(self):
        rec = self.records_cwgan[0]
        return pandas.DataFrame(
            list(self.terms['param']),
            index=self.terms.index,
            columns=rec.param_element_names,
        )

    def save(self):
        self.log('Saving results...')
        out = pathlib.Path(self.results_directory)
        out.mkdir(parents=True, exist_ok=True)

        terms = self.terms.copy()
        del terms['param']
        terms.to_csv(str(out / 'terms.csv'))

        # pw1e-mean is the best, in the sense that the termination
        # rates are 100% (with some choice of termination parameter).
        # I use smooth==lookback==1 since 1 is a good number :)
        best = terms[
            (terms['gait'] == 'pw1e') &
            (terms['rolling'] == 'mean') &
            (terms['threshold'] == 0.01) &  # smallest s.t. term. rate == 100%
            (terms['smooth'] == 1) &
            (terms['lookback'] == 1)
        ]
        best_cwgan = best[best['method'] == 'GAN']
        best_moments = best[best['method'] == 'MM']

        # Make sure that termination rate is 100%:
        assert not any(best_cwgan['first_cross'].isnull())
        assert not any(best_moments['first_cross'].isnull())

        best_cwgan.to_csv(str(out / 'best_cwgan.csv'))
        best_moments.to_csv(str(out / 'best_moments.csv'))

        params = self.make_param_table()
        params.to_csv(str(out / 'gen_params.csv'))

    def load_results(self):
        out = pathlib.Path(self.results_directory)
        self.terms = pandas.read_csv(str(out / 'terms.csv'),
                                     index_col='term_id')

    @classmethod
    def load(cls, **kwargs):
        self = cls(**kwargs)
        self.load_results()
        return self

    def dump_meta_info(self):
        out = pathlib.Path(self.results_directory)
        out.mkdir(parents=True, exist_ok=True)

        import json
        info = get_meta_info()
        with open(str(out / 'meta_info.json'), 'w') as file:
            json.dump(info, file)

    @classmethod
    def cli(cls, **kwargs):
        self = cls(**kwargs)
        self.dump_meta_info()
        self.load_records()
        self.simulate_terminations()
        self.save()
        return self


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)
    parser.add_argument(
        'results_directory', default='results', nargs='?',
        help='output directory')
    ns = parser.parse_args(args)
    return TerminationAnalyzer.cli(**vars(ns))


if __name__ == '__main__':
    analyzer = main()
