NVBIT_TOOL=../../src/memleap.so

env MATRIX_DIM=512 LD_PRELOAD=${NVBIT_TOOL} ./matmul
env MATRIX_DIM=1024 LD_PRELOAD=${NVBIT_TOOL} ./matmul
env MATRIX_DIM=2048 LD_PRELOAD=${NVBIT_TOOL} ./matmul