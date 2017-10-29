.PHONY: doc ext test* prepare

PYTEST = misc/pytest $(PYTEST_OPTS)
PYTEST_OPTS ?=

prepare: env ext

ext: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=nips_madness/ext

test: prepare
	$(PYTEST)

test-slow-only: prepare
	$(PYTEST) -k 'slowtest'

test-quick: prepare
	$(PYTEST) -k 'not slowtest'

test-flakes: prepare
	$(PYTEST) -m flakes

doc: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=doc html

include misc/conda.mk
include misc/rc/setup.mk
