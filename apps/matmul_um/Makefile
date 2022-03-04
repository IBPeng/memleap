CC := $(CUDA_HOME)/bin/nvcc
CUDA_LIB_DIR := $(CUDA_HOME)/lib64
CFLAGS := -gencode arch=compute_70,code=sm_70
INCLUDE := .
SRC := main.cu util.cu
OBJ := $(SRC:.cu=.o)
EXE := matmul

#NVBIT_INC ?= ../../../tools/data_movement
#NVBIT_PATH=$(abspath $(NVBIT_INC))

#ifeq ($(UM_PROF),)
#	NVBIT_LIB=$(NVBIT_PATH)/data_movement.so
#else
#	NVBIT_LIB=$(NVBIT_PATH)/nvbit_um.o
#	CFLAGS += -DCUPTI_PROFILING
#endif

.PHONY: clean test run_oversub profile trace

%.o:%.cu
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $< -I$(INCLUDE)

$(EXE): clean $(OBJ) util.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $(EXE) $(OBJ) -L$(CUDA_LIB_DIR) -lcuda

test: $(EXE)
	env MATRIX_DIM=1024 ./$(EXE)

run_oversub: $(EXE)
	env MATRIX_DIM=40960 ./$(EXE)

trace: $(EXE)
	env MATRIX_DIM=4096  nvprof --print-pci-topology --print-nvlink-topology --unified-memory-profiling per-process-device --print-gpu-trace ./$(EXE)

profile: $(EXE)
	env MATRIX_DIM=4096  nvprof --print-pci-topology --print-nvlink-topology --unified-memory-profiling per-process-device ./$(EXE)

clean: $(SRC)
	rm -f $(EXE) *.o *~
