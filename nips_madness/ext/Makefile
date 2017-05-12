CFLAGS ?= -fPIC -Wall -O3

libssnode.so: ssnode.c
	$(CC) -std=c99 -fopenmp $(CFLAGS) $(CPPFLAGS) -shared $< -o $@ -lgsl -lgslcblas
