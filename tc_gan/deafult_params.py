import numpy as np
DEFAULT_PARAMS = dict(
    N=102,
    J=np.array([[.0957, .0638], [.1197, .0479]]),
    D=np.array([[.7660, .5106], [.9575, .3830]]),
    S=np.array([[.6667, .2], [1.333, .2]]) / 8,
    bandwidths=[0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1],
    smoothness=0.25/8,
    contrast=[20],
    offset=[0],
    io_type='asym_tanh',
    k=0.01,
    n=2.2,
    rate_soft_bound=200, rate_hard_bound=1000,
    tau=(0.01589, 0.002),  # Superstition: integer ratio tau_E/I is bad.
)
