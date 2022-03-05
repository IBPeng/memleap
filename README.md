# MemLeap
MemLeap is a GPU Memory Profiler built atop Nvbit (NVidia Binary Instrumentation Tool) framework. It analyzes memory accesses to memory objects in GPU memory, including Unified Memory, to optimize memory usage. 

## Quick Start

1. Compile the memleap tool (memleap.so):

```bash
cd src
make
```

2. Compile an application (4 example applications are provided in /apps) to be profiled:

BFS
```bash
cd apps/bfs_um
make
```
XSBench
```bash
cd apps/XSBench/cuda_um
make
```
matmul
```bash
cd apps/matmul_um
make
```
LULESH
```bash
cd apps/LULESH/cuda/src
make
```

3. Run an application natively:

```bash
bash run.sh
```

4. Profile an application:

```bash
bash profile.sh
```

## Example Output

Some Example output can be found:

```bash
cat ./apps/XSBench/cuda_um/output_profiling_xs.log
cat ./apps/bfs_um/output_profiling_bfs.log
cat ./apps/matmul_um/output_profiling_matmul.log
cat ./apps/LULESH/cuda/src/output_profiling_lulesh.log
```

## Limitatoins

Profiling may hangs for large problems, which may be resolved by tuning the number of channels and buffer size in config.h
Currently profiling multi-GPU runs is not supported. 



