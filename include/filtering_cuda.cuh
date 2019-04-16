/* DOCUMENTATION

*/



#ifndef _FILTERING_CUDA_H_
#define _FILTERING_CUDA_H_



#include "inv_index.hpp"



/* DOCUMENTATION
 *
 */
__global__
void filtering_kernel_cuda (
    unsigned int* tokens, unsigned int* pos,     unsigned int* len,
	entry_t*      lists,  unsigned int* pos_idx, unsigned int* len_idx,
	short* scores, float threshold, unsigned int n
);



/* DOCUMENTATION
 *
 */
__host__
double filtering_cuda (
    unsigned int* tokens, unsigned int* pos, unsigned int* len,
    entry_t* lists, unsigned int* pos_idx, unsigned int* len_idx,
    float threshold, unsigned int n_sets,

    unsigned int* d_comp_buckets,
    short*        d_partial_scores,
    int*          d_nres,
    unsigned int* comp_buckets_size_out

    // short** partial_scores_out,
    // unsigned int** candidates,
    // unsigned int* candidates_size
);



#endif // _FILTERING_CUDA_H_
