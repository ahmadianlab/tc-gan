from ctypes import c_int, c_double
import ctypes
import os

import numpy

libdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext')


def load_library(name):
    return numpy.ctypeslib.load_library(name, libdir)

double_ptr = ctypes.POINTER(ctypes.c_double)

libssnode = load_library('libssnode')
for fun in [libssnode.solve_dynamics_asym_linear_gsl,
            libssnode.solve_dynamics_asym_tanh_gsl,
            libssnode.solve_dynamics_asym_linear_euler,
            libssnode.solve_dynamics_asym_tanh_euler]:
    fun.argtypes = [
        c_int, double_ptr, double_ptr, c_double, c_double,
        double_ptr, double_ptr,
        c_double, c_double,
        c_double, c_int, c_double,
        c_double, c_double,
    ]
    fun.restype = ctypes.c_int
