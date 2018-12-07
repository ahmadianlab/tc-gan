FROM nvidia/cuda

# Install system dependency
RUN apt-get update -y
RUN apt-get install -y wget

# Install Miniconda3
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/miniconda

# Install dependencies using conda and pip
RUN mkdir -pv /opt/tc-gan/misc/rc
WORKDIR /opt/tc-gan

# Copy files required only for environment setup.  This way, conda and pip
# will not be run unless these files are changed:
COPY Makefile requirements-*.txt ./
COPY misc/conda.mk misc/
COPY misc/rc/setup.mk misc/rc/
RUN make configure-default env CONDA=/opt/miniconda/bin/conda

# Finally copy everything
COPY . .
