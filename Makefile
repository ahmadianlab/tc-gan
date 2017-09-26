.PHONY: ext test prepare

prepare: ext env

ext:
	$(MAKE) --directory=nips_madness/ext

test:
	${PWD}/env/bin/pytest nips_madness

include misc/conda.mk
