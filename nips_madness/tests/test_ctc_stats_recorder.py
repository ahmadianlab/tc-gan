import numpy as np

from ..recorders import ConditionalTuningCurveStatsRecorder
from ..utils import cartesian_product
from ..networks.tests.test_conditional_minibatch import arangemd


def test_ctc_analyze_rect_data():
    contrasts = [5, 20]
    probe_offsets = [0, 0.5]
    cell_types = [0, 1]
    conditions = cartesian_product(contrasts, probe_offsets, cell_types).T
    num_conditions = len(conditions)

    num_bandwidths = 5
    tuning_curves = tc_mean = arangemd((len(conditions), num_bandwidths))

    count = 3
    conditions = np.tile(conditions, (count, 1))
    tuning_curves = np.tile(tuning_curves, (count, 1))

    indices = np.arange(len(conditions))
    np.random.RandomState(0).shuffle(indices)
    tuning_curves = tuning_curves[indices]
    conditions = conditions[indices]

    table = list(ConditionalTuningCurveStatsRecorder.analyze(tuning_curves,
                                                             conditions))
    table = np.array(table)

    recorder = ConditionalTuningCurveStatsRecorder(None, num_bandwidths)
    shift_contrast = recorder.column_names.index('contrast')
    ncols = len(recorder.column_names[shift_contrast:])
    assert table.shape == (num_conditions, ncols)

    i_count = recorder.column_names.index('count') - shift_contrast
    i_mean_beg = recorder.column_names.index('mean_0') - shift_contrast
    i_mean_end = recorder.column_names.index('var_0') - shift_contrast
    i_var_beg = i_mean_end
    i_var_end = None

    np.testing.assert_equal(table[:, i_count], count)
    np.testing.assert_equal(table[:, i_var_beg:i_var_end], 0)
    np.testing.assert_equal(table[:, i_mean_beg:i_mean_end], tc_mean)
