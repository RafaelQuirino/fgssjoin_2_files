#include "util_cuda.cuh"

int WARP_SIZE = 32;

void get_grid_config (dim3 &grid, dim3 &threads)
{
    //Get the device properties
    static bool flag = 0;
    static dim3 lgrid, lthreads;
    if (!flag) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);

		//Adjust the grid dimensions based on the device properties
		int num_blocks = 1024 * 2 * devProp.multiProcessorCount;
		lgrid = dim3(num_blocks);
		lthreads = dim3(devProp.maxThreadsPerBlock / 4);
		flag = 1;
    }
    grid = lgrid;
    threads = lthreads;
}

void __gpuAssert (cudaError_t stat, int line, std::string file) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "Error, %s at line %d in file %s\n",
            cudaGetErrorString(stat), line, file.data());
        exit(1);
    }
}

/*
__device__ float atomicAddFloat (float* address, float val)
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
//*/

/*
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
//*/
