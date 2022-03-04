# UMBench

#### Requirements: 
Unified Memory support, SM_60, SM_70

CUDA 10.0 and above 

#### Compile: 
```.bash
make
```

#### Test: 
```.bash
make test
```

#### Run with different matrix dimensions: 
```.bash
env MATRIX_DIM=XXXX ./umtest
```

#### Run an example with memory allocation > GPU memory capacity: 
```.bash
make run_oversub
```

#### Run with profiling : 
```.bash
make profile
```

