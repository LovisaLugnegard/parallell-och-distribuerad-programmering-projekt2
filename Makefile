CC=gcc
CFLAGS=-Wall -O3
LIBS=-lm

wave: wave.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
