#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cstdio>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

extern int WARP_SIZE;

void get_grid_config(dim3 &grid, dim3 &threads);
void __gpuAssert(cudaError_t stat, int line, std::string file);
#define gpuAssert(value)  __gpuAssert((value),(__LINE__),(__FILE__))
#define gpu(value)  __gpuAssert((value),(__LINE__),(__FILE__))


//__device__ float atomicAddFloat (float* address, float val);
//__device__ double atomicAdd (double* address, double val);

/*
__device__ inline float atomicAddFloat (float* address, float val)
{
    int* address_as_ull = (int*)address;
    int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __float_as_int(val +
                               __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}
*/

#endif /* UTIL_CUH_ */
