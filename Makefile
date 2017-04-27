CC=mpicc
CCFLAGS = -03
CCGFLAGS=-g
CFLAGS=-Wall -O3
LIBS=-lmpi -lm

BINS = wave

wave: wave.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
