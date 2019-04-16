/* DOCUMENTATION

*/



#ifndef _CHECKING_CUDA_CUH_
#define _CHECKING_CUDA_CUH_



/* DOCUMENTATION
 *
 */
__global__
void checking_kernel_cuda (
	unsigned int* buckets, unsigned short* scores, short* partial_scores,
    unsigned int* tokens, unsigned int* pos, unsigned int* len,
    float threshold, unsigned int n_sets, unsigned int csize
);



/* DOCUMENTATION

*/
__host__
double checking_cuda (
    unsigned int* d_tokens,  unsigned int* d_pos,     unsigned int* d_len,
	unsigned int* d_buckets, short* d_partial_scores, int*          d_nres,
    float threshold,         unsigned int csize,      unsigned int  n_sets,
	unsigned int**   similar_pairs_out,
    unsigned short** scores_out,
    int*             num_pairs_out
);



#endif // _CHECKING_CUDA_CUH_
