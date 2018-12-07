.PHONY: doc ext test* prepare

PYTEST = misc/pytest $(PYTEST_OPTS)
PYTEST_OPTS ?=

prepare: env ext

ext: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=tc_gan/ext

test: prepare
	$(PYTEST) -k 'not slowtest'
	THEANO_FLAGS=mode=Mode $(PYTEST) -k 'slowtest' \
		--junitxml=test-results/pytest-slow.xml
# See also [[./.circleci/config.yml::name: run tests]]

test-slow-only: prepare
	$(PYTEST) -k 'slowtest'

test-quick: prepare
	$(PYTEST) -k 'not slowtest'

test-flakes: prepare
	$(PYTEST) -m flakes

test-old-gan: prepare
	TEST_OLD_GAN=yes $(PYTEST)

doc: misc/rc/rc.sh env
	misc/with-env $(MAKE) --directory=doc html

doc-from-scratch:
	$(MAKE) --directory=doc clean
	$(MAKE) doc

include misc/conda.mk
include misc/rc/setup.mk
