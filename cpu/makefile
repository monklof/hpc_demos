OLD  := gemm_baseline
NEW  := gemm_8x8_blocking_pack_ab_sse
# NEW  := gemm_8x8_blocking_sse
#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -std=c99 -O2 -Wall -msse3
LDFLAGS    := -lm

UTIL       := util.o

TEST_OBJS  := test_gemm.o $(NEW).o 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_gemm.x

test_gemm.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_gemm.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
