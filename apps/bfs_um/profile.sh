NVBIT_TOOL=../../src/memleap.so

#LD_PRELOAD=${NVBIT_TOOL} ./bfs  ./input/graph4096.txt
LD_PRELOAD=${NVBIT_TOOL} ./bfs  ./input/graph65536.txt
#LD_PRELOAD=${NVBIT_TOOL} ./bfs  ./input/graph1MW_6.txt
#LD_PRELOAD=${NVBIT_TOOL} ./bfs  ./input/graph4194304.txt
