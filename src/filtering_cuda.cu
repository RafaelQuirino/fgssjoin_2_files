#include "util_cuda.cuh"
#include "filtering_cuda.cuh"
#include "compaction_cuda.cuh"
#include "similarity_cuda.cuh"

#include "util.hpp"
#include "compaction.h"



/* DOCUMENTATION
 *
 */
__global__
void filtering_kernel_cuda
(
    unsigned* tokens, unsigned int* pos,     unsigned int* len,
    entry_t*  lists,  unsigned int* pos_idx, unsigned int* len_idx,
    short* scores, float threshold, unsigned int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        unsigned int query = i;
        unsigned int query_len = len[i];

        unsigned int query_jac_prefix_size = jac_max_prefix (query_len, threshold);
        unsigned int query_jac_min_size = (unsigned int) floor(jac_min_size(query_len, threshold));
        unsigned int query_jac_max_size = (unsigned int) floor(jac_max_size(query_len, threshold));

        // For each query token in the prefix...
        for (unsigned int j = 0; j < query_jac_prefix_size; j++)
        {
            unsigned int query_tpos = j;

            unsigned int token = tokens[pos[i]+j];
            unsigned int list_size  = len_idx[token];
            unsigned int list_index = pos_idx[token];

            // For each source token in the inv_index list of query token...
            for (unsigned int k = 0; k < list_size; k++)
            {
                entry_t e = lists[list_index + k];
                unsigned int source = (unsigned int) e.doc_id;
                unsigned int source_tpos = (unsigned int) e.pos;

				if (query < source)
                {
                    unsigned int source_len = len[source];

                    // unsigned long bucket = (query * n) + source;
                    // short score = scores[ bucket ];

                    unsigned int source_jac_min_size = (unsigned int) floor(jac_min_size(source_len, threshold));
                    unsigned int source_jac_max_size = (unsigned int) floor(jac_max_size(source_len, threshold));

                    if (
                        source_len < query_jac_min_size ||
                        source_len > query_jac_max_size ||
                        query_len < source_jac_min_size ||
                        query_len > source_jac_max_size
                    )
                    {
                        // scores[ bucket ] = -1;
                        continue;
                    }

                    unsigned long bucket = (query * n) + source;
                    short score = scores[ bucket ];

                    if (score >= 0)
                    {
                        unsigned int query_rem = query_len - query_tpos;
                        unsigned int source_rem = source_len - source_tpos;
                        unsigned int min_rem = query_rem < source_rem ? query_rem : source_rem;
                        float jac_minoverlap = jac_min_overlap (query_len, source_len, threshold);

                        if ((score + 1 + min_rem) < jac_minoverlap)
                            scores[ bucket ] = -1;
                        else
                            scores[ bucket ] += 1;

                    } // if (score >= 0)
                } // if (query < set)
            } // for (unsigned k = 0; k < list_size; k++)
        } // for (unsigned j = 0; j < prefix_size; j++)
    }
}



/* DOCUMENTATION
 *
 */
__host__
double filtering_cuda 
(
    unsigned int* d_tokens,  unsigned int* d_pos,     unsigned int* d_len,
    entry_t*      d_lists,   unsigned int* d_pos_idx, unsigned int* d_len_idx,
    float         threshold, unsigned int  n_sets,
    unsigned int* d_comp_buckets,
    short*        d_partial_scores,
    int*          d_nres,
    unsigned int* comp_buckets_size_out
)
{
    unsigned long t0, t1, t00, t01;

    fprintf(stderr, "=> FILTERING...\n");
    t00 = ut_get_time_in_microseconds();

    dim3 grid, block;
    get_grid_config(grid, block);

    unsigned int bsize = n_sets * n_sets;

    // SETTING GPU PARTIAL SCORES TO 0 -----------------------------------------
    fprintf(stderr, "\t* Setting d_partial_scores memory...\n");
    t0 = ut_get_time_in_microseconds();

    gpu( cudaMemset (d_partial_scores, 0, bsize * sizeof(short)));

    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));
    //--------------------------------------------------------------------------

    // CALLING THE FILTERING KERNEL --------------------------------------------
    fprintf(stderr, "\t* Calling filtering_kernel, 5th version...\n");
    t0 = ut_get_time_in_microseconds();

    filtering_kernel_cuda <<<grid,block>>> (
        d_tokens, d_pos, d_len, d_lists, d_pos_idx, d_len_idx,
        d_partial_scores, threshold, n_sets
    );
    gpu(cudaDeviceSynchronize());

    t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done. It took %g ms.\n", ut_interval_in_miliseconds (t0, t1));
    //--------------------------------------------------------------------------

    // COMPACTING FILTERED CANDIDATES ------------------------------------------
    fprintf(stderr, "\t* Compacting filtered buckets...\n");
    t0 = ut_get_time_in_microseconds();

    gpuAssert(cudaMemset(d_nres, 0, sizeof(int)));

    filter_k_cuda <<<grid,block>>> (
        d_comp_buckets, d_partial_scores, d_nres, bsize
    );
    gpu(cudaDeviceSynchronize());

    int nres;
    gpuAssert(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));

    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));
    //--------------------------------------------------------------------------

    unsigned int comp_buckets_size = (unsigned int) nres;

    *comp_buckets_size_out = comp_buckets_size;

    fprintf (stderr, "\t# NUMBER OF CANDIDATE PAIRS: %d\n", nres);

    t01 = ut_get_time_in_microseconds();
    fprintf (stderr, "DONE IN %g ms.\n\n", ut_interval_in_miliseconds (t00,t01));

    return ut_interval_in_miliseconds (t00,t01);
}
