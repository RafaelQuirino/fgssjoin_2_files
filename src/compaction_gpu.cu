#include "compaction_gpu.cuh"



/* DOCUMENTATION
 *
 */
__global__
void filter_k_cuda (unsigned int *dst, const short *src, int *nres, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = i;
    }
}



/* DOCUMENTATION
 *
 */
__global__
void filter_k_2_cuda (
    unsigned short *dst1, unsigned short *src1,
    unsigned int   *dst2, unsigned int *src2,
    int *dstsize, int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        if (src1[i] > 0) {
            int pos = atomicAdd(dstsize, 1);
            dst1[pos] = src1[i];
            dst2[pos] = src2[i];
        }
    }
}



/* DOCUMENTATION
 *
 */
__global__
void filter_k_int_cuda (unsigned int *dst, const int *src, int *nres, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = i;
    }
}



/* DOCUMENTATION
 *
 */
__global__
void filter_k_short_cuda (short *dst, const short *src, int *nres, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = i;
    }
}



/* DOCUMENTATION
 *
 */
#define WARP_SZ 32
__device__
inline int lane_id (void) 
{ 
    return threadIdx.x % WARP_SZ; 
}



/* DOCUMENTATION
 * Warp-aggregated atomic increment
 */
__device__
int atomicAggInc (int *ctr)
{
    // int mask = __ballot(1);
    int mask = __ballot_sync(1, true);

    // select the leader
    int leader = __ffs(mask) - 1;
    // leader does the update
    int res;
    if (lane_id() == leader)
        res = atomicAdd(ctr, __popc(mask));

    // broadcast result
    // res = __shfl(res, leader);
    res = __shfl_sync(res, leader, true);

    // each thread computes its own value
    return res + __popc(mask & ((1 << lane_id()) - 1));
} // atomicAggInc



/* DOCUMENTATION
 *
 */
__global__
void warp_agg_filter_k_cuda (
    unsigned int *dst, const short *src, int* nres, int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        if (src[i] > 0)
            dst[atomicAggInc(nres)] = i;
    }
}
