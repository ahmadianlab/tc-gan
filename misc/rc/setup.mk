.PHONY: configure-*

misc/rc/rc.sh:
	@echo 'Run one of the following configuration commands first:'
	@echo '  make configure-default'
	@echo '  make configure-talapas'
	@false

configure-default:
	rm -f misc/rc/rc.sh
	touch misc/rc/rc.sh

configure-talapas configure-bitbucket: configure-%:
	ln --symbolic --force rc-$*.sh misc/rc/rc.sh
