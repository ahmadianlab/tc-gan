CFLAGS ?= -std=c99 -fPIC -Wall -O3

libssnode.so: ssnode.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -shared $< -o $@
