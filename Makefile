.PHONY: doc ext test* prepare

prepare: env ext

ext: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=nips_madness/ext

test: prepare
	misc/pytest

test-quick: prepare
	misc/pytest -k 'not slowtest'

test-flakes: prepare
	misc/pytest -m flakes

doc: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=doc html

include misc/conda.mk
include misc/rc/setup.mk
