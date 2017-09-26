# SSN-GAN simulator


## Summary of entry points

```sh
make               # Prepare everything required for simulations
make test          # Run unit tests
make env-update    # Update conda environment
./run <PATH.TO.PYTHON.MODULE> [-- ARGUMENTS]
```


## Requirements

- `conda`
- `gcc` or `icc`


## Preparation

Just run:
```
make
```

This should prepare everything required for simulations, including
installation of the relevant packages (such as Theano and Lasagne) and
compilation of the C modules.  See below for more information.


### Compiling C code in the cluster node

C code can be compiled by just running `make ext` at the root of this
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


### Setup conda environment

- Command `make env` creates conda environment and install packages
  listed in `requirements-conda.txt` and `requirements-pip.txt`.

- Command `make env-update` re-installs packages listed in
  `requirements-conda.txt` and `requirements-pip.txt`.  Run this
  command if one of `requirements-*.txt` files is updated.

- Packages listed in `requirements-conda.txt` are installed via
  `conda` command.


## Testing

Just run:
```
make test
```


## Some useful commands

Generate tuning curves:

```sh
./run nips_madness/analyzers/csv_tuning_curves.py -- logfiles/SSNGAN_XXX.log tcs --NZ=100
```
