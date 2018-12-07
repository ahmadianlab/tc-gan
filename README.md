# SSN-GAN simulator


## Summary of entry points

```sh
make               # Prepare everything required for simulations
make test          # Run unit tests
make test-quick    # Run unit tests, excluding slow tests
make env-update    # Update conda environment
./run <PATH.TO.PYTHON.MODULE> [-- ARGUMENTS]
```


## Requirements

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

### Building a Docker image

To build a Docker image, run

```
docker image build -t tc-gan .
```

where `tc-gan` can be an arbitrary tag.


## How to run simulations described in the paper

To produce data for `Figure $i`, look at the directory `scripts/fig$i`
and run script `run.sh` in the directories below.  Each `run.sh` can
be invoked as follows.

* `scripts/fig4/{mm,gan}/run.sh`, `scripts/fig6/{mm,gan}/*/run.sh`:
  `./run.sh [PATH/TO/PROJECT_ROOT/run]` where `PATH/TO/PROJECT_ROOT`
  is the directory with this `README.md` file.  The default first
  argument is `/opt/tc-gan/run`.


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


## Testing

Just run:
```
make test
```

### Running the test suite in Docker

```
docker run -t --rm tc-gan make test
```

where `tc-gan` is the tag used by `docker image build` in _Building a
Docker image_.


## Some useful commands

Generate tuning curves:

```sh
./run tc_gan/analyzers/csv_tuning_curves.py -- logfiles/SSNGAN_XXX.log tcs --NZ=100
```
