import multiprocessing
import os


def cpu_count(_environ=os.environ):
    """cpu_count() -> int
    Return available number of CPUs; OpenMP/Slurm/PBS-aware version.
    """

    try:
        return int(_environ['OMP_NUM_THREADS'])
    except (KeyError, ValueError):
        pass

    try:
        # Note: Using SLURM_CPUS_PER_TASK here instead of
        # SLURM_CPUS_ON_NODE and SLURM_JOB_CPUS_PER_NODE as they may
        # return just the total number of CPUs on the node, depending
        # on the plugin used (select/linear vs select/cons_res; see
        # document of SLURM_JOB_CPUS_PER_NODE in man sbatch and
        # SLURM_CPUS_ON_NODE in man srun).
        return int(_environ['SLURM_CPUS_PER_TASK'])
    except (KeyError, ValueError):
        pass
    if 'SLURM_JOB_ID' in _environ:
        # SLURM_CPUS_PER_TASK is specified only when --cpus-per-task
        # option is specified; otherwise, Slurm allocates one
        # processor per task (see document of --cpus-per-task in man
        # sbatch).
        return 1

    try:
        return int(_environ['PBS_NUM_PPN'])
    except (KeyError, ValueError):
        pass

    return multiprocessing.cpu_count()
