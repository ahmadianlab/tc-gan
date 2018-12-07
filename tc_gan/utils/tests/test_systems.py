import multiprocessing

from ..systems import cpu_count


def test_cpu_count():
    assert cpu_count(_environ={'OMP_NUM_THREADS': '-1'}) == -1
    assert cpu_count(_environ={'SLURM_CPUS_PER_TASK': '-3'}) == -3
    assert cpu_count(_environ={'SLURM_JOB_ID': None}) == 1
    assert cpu_count(_environ={'PBS_NUM_PPN': '-5'}) == -5
    assert cpu_count(_environ={}) == multiprocessing.cpu_count()
