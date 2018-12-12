FROM nvidia/cuda

# Install system dependency
RUN apt-get update -y
RUN apt-get install -y wget

# Install Miniconda3
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/miniconda

# Install dependencies using conda and pip
ARG builddir=/tmp/tc-gan
RUN mkdir -pv $builddir/misc/rc

# Environment variable TC_GAN_ENV sets the conda environment (thus
# Python interpreter) to be used in `./run` script (via
# `misc/activate.bash`).  `make prepare` command (via `misc/conda.mk`)
# also respects this variable.
ENV TC_GAN_ENV /opt/miniconda/envs/tc-gan

# Copy files required only for environment setup.  This way, conda and pip
# will not be run unless these files are changed:
COPY Makefile requirements-*.txt $builddir/
COPY misc/conda.mk $builddir/misc/
COPY misc/rc/setup.mk $builddir/misc/rc/
RUN make -C $builddir configure-default env CONDA=/opt/miniconda/bin/conda

ENV PATH=$TC_GAN_ENV/bin:$PATH

# Setup and switch user
ARG TC_GAN_USER=tc-gan
ARG TC_GAN_UID=1000

# Creating home because Theano etc. needs $HOME
RUN useradd \
        --create-home \
        --shell /bin/bash \
        --uid $TC_GAN_UID \
        $TC_GAN_USER

# `/srv/tc-gan` is an empty workspace (which may be used to mount
# project repository via `./docker-run` script)
RUN mkdir -pv /srv/tc-gan && chown $TC_GAN_USER /srv/tc-gan
WORKDIR /srv/tc-gan

USER $TC_GAN_USER
