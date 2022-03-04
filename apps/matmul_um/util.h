#ifndef UMTEST_UTIL_H
#define UMTEST_UTIL_H

#include <stdio.h>

static cudaDeviceProp deviceProp;

enum AllocType {UMnaive, UMPref, HpgDmcp};
static const char *AllocString[]={"naive Unified Memory",
				  "Unified Memory with async Prefetch",
				  "Host pagable memory + Device memory copy"};
  
#define ELEMENT_TYPE float

#define ERROR(fmt, ...)							\
  {printf("Error in %s line %d :: %s () " fmt, __FILE__, __LINE__, __func__,  __VA_ARGS__);exit(13);}


#define checkRes(res)\
  {if(cudaSuccess!=res){						\
      printf("Cuda Error in %s line %d :: %s\n", __FILE__, __LINE__, cudaGetErrorName(res));\
      exit(13);}							\
  }

void allocate_matrix(void **h_ptr, void **d_ptr, size_t size, AllocType allocType);
void free_matrix(ELEMENT_TYPE *h_ptr, ELEMENT_TYPE *d_ptr, size_t size, AllocType allocType);

void initMatrix(ELEMENT_TYPE* h_ptr, ELEMENT_TYPE initValue, size_t matrixDim);
bool verifyMatrix(ELEMENT_TYPE* h_ptr, ELEMENT_TYPE refValue, size_t matrixDim);

#define BLOCK_SIZE 32

__global__ void matrixMultiplyKernel(float *C, float *A, float *B, size_t matrixDim) ;


#endif
