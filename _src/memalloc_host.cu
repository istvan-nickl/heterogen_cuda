#include "memory.h"

#define USE_CUDA_MEMALLOC 0
#define USE_WC 1

void memalloc_host(int imgWidthF, int imgHeightF, unsigned char **input, 
	               int imgWidth, int imgHeight, unsigned char **output)
{
	cudaError_t cuda_ret;

#if USE_CUDA_MEMALLOC == 0
	*input   = (unsigned char *)malloc(3*imgWidthF*imgHeightF*sizeof(unsigned char));
	*output = (unsigned char *)malloc(3*imgWidth*imgHeight*sizeof(unsigned char));
#else
	#if USE_WC == 0
		cuda_ret = cudaHostAlloc(input, 3*imgWidthF*imgHeightF*sizeof(unsigned char), cudaHostAllocDefault);
	#else
		cuda_ret = cudaHostAlloc(input, 3*imgWidthF*imgHeightF*sizeof(unsigned char), cudaHostAllocWriteCombined);
	#endif
	cuda_ret = cudaHostAlloc(output, 3*imgWidth*imgHeight*sizeof(unsigned char), cudaHostAllocDefault);

#endif
}

void memfree_host(unsigned char **input, unsigned char **output)
{
	cudaError_t cuda_ret;

#if USE_CUDA_MEMALLOC == 0
	free(*input);
	free(*output);
#else
	cuda_ret = cudaFree(input);
	cuda_ret = cudaFree(output);
#endif
}