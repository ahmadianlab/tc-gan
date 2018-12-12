TC_GAN_ENV ?= ${CURDIR}/env
export TC_GAN_ENV

CONDA = conda
CONDA_OPTS ?= --yes

CONDA_INSTALL_OPTS = \
${CONDA_OPTS} --prefix $(TC_GAN_ENV) --file requirements-conda.txt

PIP = ${TC_GAN_ENV}/bin/pip

.PHONY: conda-* pip-* env-* env

env: $(TC_GAN_ENV)

$(TC_GAN_ENV):
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
