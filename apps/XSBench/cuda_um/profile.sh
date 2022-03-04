NVBIT_TOOL=../../../src/memleap.so

LD_PRELOAD=${NVBIT_TOOL}  ./XSBench -m event -s large -l 150000 -k 1 > output_profiling_xs.log 2>&1
LD_PRELOAD=${NVBIT_TOOL}  ./XSBench -m event -s large -l 300000 -k 1 
LD_PRELOAD=${NVBIT_TOOL}  ./XSBench -m event -s large -l 1000000 -k 1
LD_PRELOAD=${NVBIT_TOOL}  ./XSBench -m event -g 11303 -k 1 -l 10000
