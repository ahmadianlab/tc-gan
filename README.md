# SSN-GAN simulator

## Compiling C code in the cluster node

C code can be compiled by just running `make` at the root of this
repository.  In the cluster machines, it's better to load latest
version of the compiler.

For example, run the following to use GNU C compiler 6.1:
```
module load gcc/6.1
make -B CC=gcc
```
or to use Intel compiler:
```
module load intel/16
make -B CC=icc
```

As of writing, it seems GCC yields the faster binary.

*WARNING*: You MUST load the same compiler via the `module load ...`
command when running the Python code (e.g., `run_batch_GAN.py`).

Notes:
- `CC=gcc` (or `CC=icc`) specifies the compiler.
- `-B` (alternatively, `--always-make`) tells the `make` command to
  compile the C code even if the compiled file (`libsnnode.so`) is
  newer than the C source code.  It is useful when trying different
  compiler and compiler options.
