.PHONY: ext test prepare

prepare: env ext

ext: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=nips_madness/ext

test:
	misc/with-env pytest

test-quick:
	misc/with-env pytest -k 'not slowtest'

include misc/conda.mk
include misc/rc/setup.mk
