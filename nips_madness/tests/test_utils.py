import multiprocessing

from ..utils import cpu_count


def test_cpu_count():
    assert cpu_count(_environ={'SLURM_CPUS_ON_NODE': '-3'}) == -3
    assert cpu_count(_environ={'PBS_NUM_PPN': '-5'}) == -5
    assert cpu_count(_environ={}) == multiprocessing.cpu_count()
