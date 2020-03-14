CC=clang
CFLAGS=-std=c99 -O3 -march=native -mtune=native -fopenmp 

PETSC_DIR=/usr/local/petsc
PETSC_ARCH=arch-linux2-c-debug
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

.PHONY: all mixed seis_test seir_test

all: mixed seis_test seir_test

mixed: mixed.c
	$(CC) -shared -fPIC $(CFLAGS) $^ -o libmixed.so $(PETSC_CC_INCLUDES)

seis_test: seis_test.c
	$(CC) $(CFLAGS) $^ -o seis_test $(PETSC_CC_INCLUDES) $(PETSC_WITH_EXTERNAL_LIB) -L./ -lmixed

seir_test: seir_test.c
	$(CC) $(CFLAGS) $^ -o seir_test $(PETSC_CC_INCLUDES) $(PETSC_WITH_EXTERNAL_LIB) -L./ -lmixed
