FROM nvidia/cuda

# Install system dependency
RUN apt-get update -y
RUN apt-get install -y wget

# Install Miniconda3
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/miniconda

# Install dependencies using conda and pip
COPY . /opt/tc-gan
WORKDIR /opt/tc-gan
RUN make configure-default prepare CONDA=/opt/miniconda/bin/conda
