CUPTI_HOME = $(CUDA_HOME)/extras/CUPTI
INCLUDES = -I$(CUPTI_HOME)/include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

EXTRAS_LIB_PATH := $(CUDA_HOME)/lib64
LIB_PATH ?= $(CUPTI_HOME)/lib64

LIBS = -L $(EXTRAS_LIB_PATH)
LIBS += -lcuda -L $(LIB_PATH) -lcupti

OBJ = o
EXE = cupti_test

cupti_test: cupti_test.$(OBJ)
	nvcc $(NVCC_COMPILER) -o $@ cupti_test.$(OBJ) $(LIBS) $(INCLUDES)

cupti_test.$(OBJ): cupti_test.cu
	nvcc $(NVCC_COMPILER) -c $(INCLUDES) $<

run: cupti_test
	env LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(LIB_PATH) ./$<

clean:
	rm -f cupti_test cupti_test.$(OBJ) *~

