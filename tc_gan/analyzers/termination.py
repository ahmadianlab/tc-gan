import itertools
import pathlib

import numpy

from .learning import smape
from . import gaiting
from ..utils import Namespace, first_sign_change, make_progressbar


def terminated(rec, lookback, smooth, threshold, rolling='mean',
               average_window=None, gait_func=gaiting.smr1e):
    if average_window is None:
        average_window = lookback
    result = Namespace(**locals())

    gp = rec.flatten_gen_params()
    lookback_step = int(lookback / rec.rc.gen_learning_rate)
    smooth_step = int(smooth / rec.rc.gen_learning_rate)
    window_step = int(average_window / rec.rc.gen_learning_rate)

    diffs = gait_func(gp, lookback_step, smooth_step, rolling)
    diffs /= lookback
    i_cross = first_sign_change(diffs - threshold)
    if i_cross is None:
        result.first_cross = None
        result.param = numpy.empty_like(gp[0])
        result.param[:] = numpy.nan
        result.cost_norm2 = numpy.nan
        result.cost_smape = numpy.nan
    else:
        result.first_cross = first_cross = i_cross + lookback_step
        result.param = gp[first_cross - window_step: first_cross].mean(axis=0)
        p_gen = result.param
        p_true = rec.flatten_true_params()
        result.cost_norm2 = numpy.linalg.norm(p_gen - p_true)
        result.cost_smape = smape(p_gen, p_true)
    return result


def sweep_terminators(records, gait_funcs=[gaiting.smr1e, gaiting.pw1e],
                      lookbacks=[2, 1, 0.5], smooths=[2, 1, 0.5],
                      thresholds=[0.05, 0.01, 0.005, 0.001],
                      rollings=['mean']):
    names = ['gait_func', 'lookback', 'smooth', 'threshold', 'rolling']
    axes = [gait_funcs, lookbacks, smooths, thresholds, rollings]
    for rec in records:
        rec_id = str(rec.datastore.directory)
        for values in itertools.product(*axes):
            row = kwargs = dict(zip(names, values))
            result = terminated(rec, **kwargs)
            row['rec_id'] = rec_id
            row.update(vars(result))
            row['gait'] = row.pop('gait_func').__name__
            del row['rec']  # save memory, in case `records` is lazy
            yield row


def simulate_terminations(records=None, glob=None, progress=False, **kwargs):
    import pandas
    if records is None:
        assert glob is not None
        from ..loaders import load_records
        paths = make_progressbar(not progress)(list(pathlib.Path().glob(glob)))
        records = map(load_records, paths)
    else:
        records = make_progressbar(not progress)(records)
    return pandas.DataFrame(sweep_terminators(records, **kwargs))


def simulate_terminations_snr(*args,
                              gait_funcs=[gaiting.pw1e], rollings=['snr'],
                              thresholds=[1.0, 0.75, 0.5, 0.25],
                              **kwargs):
    """
    Same as `simulate_terminations` but with rolling SNR.

    Options are default to the values that work well with rolling SNR.
    """
    return simulate_terminations(*args,
                                 gait_funcs=gait_funcs,
                                 rollings=rollings,
                                 thresholds=thresholds,
                                 **kwargs)
