CFLAGS ?= -fPIC -Wall -O3
CC ?= gcc

libssnode.so: ssnode.c
	$(CC) -std=c99 -fopenmp \
$(CFLAGS) $(CPPFLAGS) -shared $< -o $@
