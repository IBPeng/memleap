#include "util.h"

void allocate_matrix(void **h_ptr, 
		     void **d_ptr,
		     size_t size,
		     AllocType allocType)
{
  switch(allocType){

  case UMnaive:
    checkRes(cudaMallocManaged(d_ptr, size, cudaMemAttachGlobal));
    *h_ptr = *d_ptr;
    break;
    
  case UMPref:
    checkRes(cudaMallocManaged(d_ptr, size, cudaMemAttachGlobal));
    if (deviceProp.concurrentManagedAccess) {
      checkRes(cudaMemPrefetchAsync(d_ptr, size, cudaCpuDeviceId));
    }
    *h_ptr = *d_ptr;
    break;

  case HpgDmcp:
    {
      *h_ptr = malloc(size);
      if (!(*h_ptr)) {
	ERROR("Failed to malloc on host (%zu bytes)\n", size);
      }

      checkRes(cudaMalloc(d_ptr, size));
      break;
    }
    
  default:
    ERROR("Invalid allocation type %d\n", allocType);

  }

}

void free_matrix(ELEMENT_TYPE *h_ptr, 
		 ELEMENT_TYPE *d_ptr,
		 size_t size,
		 AllocType allocType)
{
  switch(allocType){
  case UMnaive:
  case UMPref:
      checkRes(cudaFree(d_ptr));
    break;

  case HpgDmcp:
    free(h_ptr);
    checkRes(cudaFree(d_ptr));
    break;

  default:
    ERROR("Invalid allocation type %d\n", allocType);

  }

}

void initMatrix(ELEMENT_TYPE* h_ptr,
		ELEMENT_TYPE initValue,
		size_t matrixDim)
{
    for (size_t i = 0; i < matrixDim; ++i) {
    for (size_t j = 0; j < matrixDim; ++j) {
      h_ptr[j + i * matrixDim] = initValue;
    }
  }
}


bool verifyMatrix(ELEMENT_TYPE* h_ptr,
		  ELEMENT_TYPE refValue,
		  size_t matrixDim)
{
    for (size_t i = 0; i < matrixDim; ++i) {
    for (size_t j = 0; j < matrixDim; ++j) {
      if(h_ptr[j + i * matrixDim] != refValue){
	printf("h_ptr[%zu]=%f != %f \n", (j + i * matrixDim), h_ptr[j + i * matrixDim], refValue);
	return false;
      }
    }
    }
    return true;
}


__global__ void matrixMultiplyKernel(float *C, float *A, float *B,
                                     size_t matrixDim) {
  // Block index
  size_t bx = blockIdx.x;
  size_t by = blockIdx.y;

  // Thread index
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;

  size_t wA = matrixDim;
  size_t wB = matrixDim;

  // Index of the first sub-matrix of A processed by the block
  size_t aBegin = matrixDim * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  size_t aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  size_t aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  size_t bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  size_t bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (size_t a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  size_t c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}
