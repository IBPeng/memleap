NVCC=nvcc -ccbin=$(CXX) -g -D_FORCE_INLINES

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=../nvbit_release/core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit 
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)
ARCH=70 #35

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=memleap.so

BASE_PATH=../
TEST_DIR=$(BASE_PATH)/tests
RODINIA_DIR=$(TEST_DIR)/rodinia_3.1
RODINIA_DATA_DIR=$(RODINIA_DIR)/data

all: $(NVBIT_TOOL)

run: $(NVBIT_TOOL)
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/vectoradd/vectoradd 2000000
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/cuda_samples/0_Simple/simpleStreams/simpleStreams
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(TEST_DIR)/cuda_samples/6_Advanced/FDTD3d/FDTD3d --dimx=96 --dimy=96 --dimz=188
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(TEST_DIR)/cuda_samples/6_Advanced/FDTD3d/FDTD3d --dimx=96 --dimy=188 --dimz=188
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/cuda_samples/7_CUDALibraries/conjugateGradient/conjugateGradient_4096
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/cuda_samples/7_CUDALibraries/conjugateGradient/conjugateGradient_8192
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/lavaMD/lavaMD -boxes1d 6
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/lavaMD/lavaMD -boxes1d 8 
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/lavaMD/lavaMD -boxes1d 10
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/backprop/backprop 65536
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/backprop/backprop 131072
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/backprop/backprop 262144
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/backprop/backprop 524288
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/backprop/backprop 1000000
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/cfd/euler3d_double $(RODINIA_DATA_DIR)/cfd/fvcorr.domn.097K
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/lud/cuda/lud_cuda -s 1024
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/lud/cuda/lud_cuda -s 2048	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/miniFE/cuda/src/miniFE.x -nx 10 -ny 10 -nz 10
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/miniFE/cuda/src/miniFE.x -nx 20 -ny 20 -nz 20
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/miniFE/cuda/src/miniFE.x -nx 40 -ny 40 -nz 40
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/miniFE/cuda/src/miniFE.x -nx 80 -ny 80 -nz 80
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/miniFE/cuda/src/miniFE.x -nx 100 -ny 100 -nz 100
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/kmeans/kmeans -o -i $(RODINIA_DATA_DIR)/kmeans/kdd_cup	

	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/Quicksilver/src/qs -i $(TEST_DIR)/Quicksilver/Examples/CORAL2_Benchmark/Problem1/Coral2_P1.inp -X 16  -Y 16  -Z 16  -x 16  -y 16  -z 16  -I 1  -J 1  -K 1  -n 163840
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/cuda_samples/7_CUDALibraries/conjugateGradient/conjugateGradient
	#LD_PRELOAD=./$(NVBIT_TOOL) $(TEST_DIR)/cuda_samples/6_Advanced/FDTD3d/FDTD3d
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/kmeans/kmeans -o -i $(RODINIA_DATA_DIR)/kmeans/kdd_cup	
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/huffman/pavle $(RODINIA_DATA_DIR)/huffman/test1024_H2.206587175259.in
	#env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/collab/usr/global/tools/cudnn/blueos_3_ppc64le_ib/cudnn-7.6.5/lib LD_PRELOAD=./$(NVBIT_TOOL)  $(TEST_DIR)/DeepBench/code/nvidia/bin/sparse_bench
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(TEST_DIR)/DeepBench/code/nvidia/bin/gemm_bench
	#env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/collab/usr/global/tools/cudnn/blueos_3_ppc64le_ib/cudnn-7.6.5/lib LD_PRELOAD=./$(NVBIT_TOOL)  $(TEST_DIR)/DeepBench/code/nvidia/bin/conv_bench


	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/srad/srad_v2/srad 2048 2048 0 127 0 127 0.5 2
	#LD_PRELOAD=./$(NVBIT_TOOL)  $(RODINIA_DIR)/cuda/srad/srad_v1/srad 100 0.5 502 458


$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=sm_$(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=sm_$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o *~ 
