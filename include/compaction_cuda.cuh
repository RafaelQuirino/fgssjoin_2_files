/* DOCUMENTATION

*/



#ifndef _COMPACTION_CUH
#define _COMPACTION_CUH



#include "util.hpp"
#include "util_cuda.cuh"



/* DOCUMENTATION
 *
 */
__global__
void filter_k_cuda (unsigned int *dst, const short *src, int *nres, int n);



/* DOCUMENTATION
 *
 */
__global__
void filter_k_2_cuda (
    unsigned short *dst1, unsigned short *src1,
    unsigned int   *dst2, unsigned int *src2,
    int *dstsize, int n
);



/* DOCUMENTATION
 *
 */
__global__
void filter_k_int_cuda (unsigned int *dst, const int *src, int *nres, int n);



/* DOCUMENTATION
 *
 */
__global__
void filter_k_short_cuda (short *dst, const int *src, int *nres, int n);



/* DOCUMENTATION
 *
 */
__global__
void warp_agg_filter_k_cuda (
    unsigned int *dst, const short *src, int* nres, int n
);



#endif
