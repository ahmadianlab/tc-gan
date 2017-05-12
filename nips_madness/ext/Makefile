CFLAGS ?= -fPIC -Wall -O3

libssnode.so: ssnode.c gsl
	$(CC) -std=c99 -fopenmp \
$(shell gsl/bin/gsl-config --cflags --libs) \
$(CFLAGS) $(CPPFLAGS) -shared $< -o $@

gsl-2.3.tar.gz:
	wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.3.tar.gz

gsl-2.3: gsl-2.3.tar.gz
	tar xzf gsl-2.3.tar.gz
	touch $@

gsl: gsl-2.3
	cd $^ && ./configure --prefix=$$PWD/../gsl
	$(MAKE) --directory=$^
	$(MAKE) --directory=$^ install
	touch $@
