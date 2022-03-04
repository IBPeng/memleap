CC := $(CUDA_HOME)/bin/nvcc
CFLAGS := -gencode arch=compute_70,code=sm_70
INCLUDE := $(CUDA_HOME)/include
CUDA_LIB_DIR := $(CUDA_HOME)/lib64
SRC := bfs.cu
EXE := bfs


release: clean $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda

cupti: clean $(SRC)
	$(CC) $(CFLAGS) -DCUPTI_PROFILING -I../../memleap/cupti_um -I$(CUDA_HOME)/extras/CUPTI/include $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(CFLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt *~ *.o
