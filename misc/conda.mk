CONDA = conda
CONDA_OPTS ?= --yes

CONDA_INSTALL_OPTS = \
${CONDA_OPTS} --prefix env --file requirements-conda.txt

PIP = ${PWD}/env/bin/pip

.PHONY: conda-* pip-* env-*

env:
	${CONDA} create ${CONDA_INSTALL_OPTS}
	${MAKE} pip-install

env-update:
	${MAKE} env  # will not be run if env/ is there
	${MAKE} conda-install
	${MAKE} pip-install
# Note: These jobs have to be run by this order.

conda-install:
	${CONDA} install ${CONDA_INSTALL_OPTS}

pip-install:
	${PIP} install --requirement requirements-pip.txt
