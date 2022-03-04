NVBIT_TOOL=../../../../src/memleap.so

LD_PRELOAD=${NVBIT_TOOL} ./lulesh -s 64 3
LD_PRELOAD=${NVBIT_TOOL} ./lulesh -s 128 3
LD_PRELOAD=${NVBIT_TOOL} ./lulesh -s 144 10 
