#include <iostream>
#include <vector>
#include "util.h"

#ifdef CUPTI_PROFILING
#include "cupti_um.cuh"
#endif

#define VERIFY  1
//#define PREFETCH_GPU 1
#define DEFAULT_DIM  512
#define DEFAULT_REPS 10
#define DEFAULT_DEVICE_ID 0
#define DEFAULT_SM_SIZE 0
#define DEFAULT_ALLOC_TYPE UMnaive //UMPref //UMnaive


int main(int argc, char** argv){

#ifdef CUPTI_PROFILING
  uvm_profiling_init();
#endif

  int matrixDim = DEFAULT_DIM;
  int num_reps = DEFAULT_REPS;
  int device_id = DEFAULT_DEVICE_ID;

  const char* env_v0 = std::getenv("MATRIX_DIM");
  if(env_v0)
    matrixDim = atoi(env_v0);
  if( matrixDim%BLOCK_SIZE != 0){
    std::cout << "MATRIX_DIM must be divisible by MATRIX_DIM " << BLOCK_SIZE << std::endl;
    return -1;
  }
  
  const char* env_v1 = std::getenv("NUM_REPS");
  if(env_v1)
    num_reps = atoi(env_v1);

  const char* env_v2 = std::getenv("DEVICE_ID");
  if(env_v2)
    device_id = atoi(env_v2);

  size_t element_size = sizeof(ELEMENT_TYPE);
  size_t matrix_size = element_size * matrixDim * matrixDim;
  size_t total_mb = matrix_size * 3 / 1024 / 1024;

  std::cout << "Total Memory (MB) : " << total_mb << std::endl;
  std::cout << "Each matrix : " << matrixDim << "x" << matrixDim << " (" <<element_size<<" bytes)"  << std::endl;

  int device_count = 0;
  checkRes(cudaGetDeviceCount(&device_count));
  std::cout << "device_count=" << device_count << std::endl;
  if(device_count<=device_id)
    ERROR("device_id (%d) >= device_count (%d)\n", device_id, device_count);

  checkRes(cudaGetDeviceProperties(&deviceProp, device_id));
  std::cout << "CUDA compute capability : " << deviceProp.major << "." <<deviceProp.minor<< std::endl;

  checkRes(cudaSetDevice(device_id));

  
  /* Allocate memory */
  ELEMENT_TYPE *h_a, *h_b, *h_c;
  ELEMENT_TYPE *d_a, *d_b, *d_c;
  AllocType allocType = DEFAULT_ALLOC_TYPE;
  std::cout << "Allocation Scheme : " << AllocString[allocType] << std::endl;
  allocate_matrix((void**)&h_a, (void**)&d_a, matrix_size, allocType);
  allocate_matrix((void**)&h_b, (void**)&d_b, matrix_size, allocType);
  allocate_matrix((void**)&h_c, (void**)&d_c, matrix_size, allocType);
  std::cout << "h_a " << h_a << ", d_a=" << d_a << std::endl;

  /* init value (first access) */
  initMatrix(h_a, 0.5f,  matrixDim);
  initMatrix(h_b, 2.0f,  matrixDim);
  initMatrix(h_c, 0.0f,  matrixDim);
  
  /* CUDA Kernel Launch Configuration */
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(matrixDim / threads.x, matrixDim / threads.y);
  size_t shared_memory_bytes = DEFAULT_SM_SIZE;
  cudaEvent_t start_event, stop_event;
  float elapsed_time;
  cudaStream_t stream;  
  checkRes(cudaEventCreate(&start_event));
  checkRes(cudaEventCreate(&stop_event));
  checkRes(cudaStreamCreate(&stream));


  /*Optional: prefetch to GPU*/  
#ifdef PREFETCH_GPU
  if(allocType==UMPref || allocType==UMnaive){
  cudaStream_t prefStream0, prefStream1,prefStream2;
  checkRes(cudaStreamCreate(&prefStream0));
  checkRes(cudaStreamCreate(&prefStream1));
  checkRes(cudaStreamCreate(&prefStream2));
  if (deviceProp.concurrentManagedAccess) {
    checkRes(cudaMemPrefetchAsync(d_a, matrix_size, device_id, prefStream0));
    checkRes(cudaMemPrefetchAsync(d_b, matrix_size, device_id, prefStream1));
    checkRes(cudaMemPrefetchAsync(d_c, matrix_size, device_id, prefStream2));
  }
  }
#endif

  if(allocType==HpgDmcp){
    checkRes(cudaMemcpy(d_a, h_a, matrix_size, cudaMemcpyHostToDevice));
    checkRes(cudaMemcpy(d_b, h_b, matrix_size, cudaMemcpyHostToDevice));
    checkRes(cudaMemcpy(d_c, h_c, matrix_size, cudaMemcpyHostToDevice));
  }
  
  /* Run Main loop */
#ifdef CUPTI_PROFILING
  uvm_profiling_start();
#endif

  checkRes(cudaEventRecord(start_event, 0)); //on NULL (0) stream 
  //matrixMultiplyKernel<<<grid, threads>>>(d_c, d_a, d_b, matrixDim);
  matrixMultiplyKernel<<<grid, threads, shared_memory_bytes, stream>>>(d_c, d_a, d_b, matrixDim);
  checkRes(cudaStreamSynchronize(stream));
  checkRes(cudaEventRecord(stop_event, 0));
  checkRes(cudaEventSynchronize(stop_event));
  checkRes(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

#ifdef CUPTI_PROFILING
  uvm_profiling_stop();
#endif
  std::cout << "Elapsed time (ms) : " << elapsed_time
	    << ", Bandwidth (GB/s) : " << total_mb/elapsed_time << std::endl;

  
#if VERIFY
  if(allocType==HpgDmcp)
    checkRes(cudaMemcpy(h_c, d_c, matrix_size, cudaMemcpyDeviceToHost));

  bool validated = verifyMatrix(h_c, (ELEMENT_TYPE) matrixDim,  matrixDim);
  std::cout << "Validation: " << (validated ?"Passed" :"Failed") << std::endl;
#endif
  
 
  free_matrix(h_a, d_a, matrix_size, allocType);
  free_matrix(h_b, d_b, matrix_size, allocType);
  free_matrix(h_c, d_c, matrix_size, allocType);
  
  return 0;
}
