# Code repository for "Inferring neural circuit structure from datasets of heterogeneous tuning curves"

[![Build Status][travis-img]][travis-url]

This package includes codes that implement the experiments Of Arakaki et al., Inferring neural circuit structure from datasets of heterogeneous tuning curves, which used Generative Adversarial Networks and Moment Matching to fit two mechanistic circuit models to tuning curve datasets.  

## Summary of entry points

```sh
make               # Prepare everything required for simulations
make test          # Run unit tests
make test-quick    # Run unit tests, excluding slow tests
make env-update    # Update conda environment
./run <PATH.TO.PYTHON.MODULE> [-- ARGUMENTS]
```


## Requirements

- `python`
- `conda`
- `gcc` or `icc`


## Preparation

After `git checkout` just run the following commands:

```
make configure-talapas    # IF in talapas
make configure-default    # otherwise
make
```

These commands have to be run only once (in principle).
This should prepare everything required for simulations, including
installation of the relevant packages (such as Theano and Lasagne) and
compilation of the C modules.  See below for more information.


## How to run trainings described in the paper

### Figure 3

Run the following command to produce data for Figure 3.

```sh
./setup.sh
python FF_lalazar_model.py 40
```

To run the last command in a Docker container, use, e.g.,

```sh
./docker-run \
    --env THEANO_FLAGS=device=cuda,floatX=float32,force_device=True -- \
    python FF_lalazar_model.py 40
```

### Figure 4 and 6

To produce data for `Figure $i`, look at the directory `scripts/fig$i`
and run script `run.sh` in the directories below.  Invoking `run.sh`
runs the training and store the result in the directory in which
`run.sh` is located.  To run it inside a Docker container, put
`./docker-run` and appropriate options in front of it, e.g.,

```sh
./docker-run \
    --env THEANO_FLAGS=device=cuda,floatX=float32,force_device=True -- \
    scripts/fig4/gan/run.sh
```

## How to analyze and plot the figures in the paper

To reproduce the visualizations of figures 3 and 4 use the scripts `Fig3analysis.py` and `Fig4analysis.py`. In `Fig4analaysis.py` you must manually set the path of the tuning curve data using the positional command line arguments. To recreate figure 5 use `Fig5analysis.py` using the positional command line arguments to set the path to the GAN and MM data you wish to use. The script assumes that the true parameters of both runs are the same. TO recreaste figure 6 use `Fig6analysis.py`. This script assumes that you have already run the `fig6` run script and all the resulting data is saved in the folders `scripts/fig6/*`.

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

- Command `make env` creates conda environment and installs packages
  listed in `requirements-conda.txt` and `requirements-pip.txt`.

- Command `make env-update` re-installs packages listed in
  `requirements-conda.txt` and `requirements-pip.txt`.  Run this
  command if one of `requirements-*.txt` files is updated.

- Packages listed in `requirements-conda.txt` are installed via
  `conda` command.


### Per-machine configuration

Some machine specific configuration is done via `misc/rc/rc.sh`.  This
file is `source`ed from all entry points.

See also:

- `misc/with-env COMMAND [ARGUMENTS]` runs `COMMAND` in a bash process
  configured with `misc/rc/rc.sh` and the conda environment.

- `misc/rc/rc-talapas.sh` is used if the repository is configured with
  `make configure-talapas`.


## Using GPU in Talapas

To use GPU in Talapas, one has to launch a job in partition `gpu` with
flag `--gres=gpu:1`.  See also:
[How-to Submit a GPU Job - Talapas Knowledge Base](https://hpcrcf.atlassian.net/wiki/spaces/TCP/pages/7289618/How-to+Submit+a+GPU+Job)

### Quick check Theano installation

Following instruction assumes that Preparation commands (see above)
are already run.

```console
[you@ln1] $ cd PATH/TO/THIS/REPOSITORY
[you@ln1] $ srun --partition gpu --gres=gpu:1 --pty bash
[you@n098] $ module load cuda
[you@n098] $ THEANO_FLAGS=device=cuda misc/with-env python -c 'import theano'
Using cuDNN version 5110 on context None
Mapped name None to device cuda: Tesla K80 (0000:04:00.0)
```

(where `ln1` is a head node and `n098` is a GPU node)


## Executing commands inside Docker

Our program can be run inside the Docker container.  We provide
`./docker-run` script to help setting up the Docker container.  You
may manually build the Docker image from `Dockerfile` in this
directory.

Examples:

* `./docker-run --dry-run`: Print commands to be executed.
* `./docker-run`: Starts an interactive bash session.
* `./docker-run -i -- ipython`: Starts an interactive IPython session.
* `./docker-run -- ./run tc_gan.run.bptt_cwgan [-- ARGUMENTS]`: Run a
  submodule `tc_gan.run.bptt_cwgan` as a script.  Replace it with any
  module with the `main` function.
* `./docker-run -- make test`: Run tests.

Notes:

* You can use `--env` flag to pass `THEANO_FLAGS`.  For example:
  `./docker-run --env THEANO_FLAGS=device=cuda,floatX=float32,force_device=True -- python FF_lalazar_model.py 40`
* `./docker-run` builds an appropriate Docker image with all the
  requirements when it is run for the first time.  No explicit build
  step is required.
* `./docker-run` mounts the project root directory as the current
  directory inside the Docker image.  Saving files to under directory
  will be reflected in the host filesystem.


## Testing

Just run:
```
make test
```


## Some useful commands

Generate tuning curves:

```sh
./run tc_gan/analyzers/csv_tuning_curves.py -- logfiles/SSNGAN_XXX.log tcs --NZ=100
```

[travis-img]: https://travis-ci.com/ahmadianlab/tc-gan.svg?branch=master
[travis-url]: https://travis-ci.org/ahmadianlab/tc-gan
