.PHONY: setup-talapas

misc/rc/rc.sh:
	@echo 'Run one of the following configuration commands first:'
	@echo '  make configure-default'
	@echo '  make configure-talapas'
	@false

configure-default:
	rm -f misc/rc/rc.sh
	touch misc/rc/rc.sh

configure-talapas:
	ln --symbolic --force rc-talapas.sh misc/rc/rc.sh
