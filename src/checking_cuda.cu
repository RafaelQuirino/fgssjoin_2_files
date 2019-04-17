#include "util.hpp"
#include "compaction.h"
#include "util_cuda.cuh"
#include "checking_cuda.cuh"
#include "similarity_cuda.cuh"
#include "compaction_cuda.cuh"



/* DOCUMENTATION
 *
 */
__global__
void checking_kernel_cuda 
(
	unsigned int* buckets, unsigned short* scores, short* partial_scores,
	unsigned int* tokens, unsigned int* pos, unsigned int* len,
	float threshold, unsigned int n_sets, unsigned int csize
)
{
	unsigned int n = csize;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = idx; i < n; i += blockDim.x * gridDim.x)
	{
		unsigned int bucket = buckets[i];

		unsigned int query  = bucket / n_sets;
		unsigned int source = bucket % n_sets;
		unsigned int query_len  = len[query];
		unsigned int source_len = len[source];
		float minoverlap = jac_min_overlap(query_len, source_len, threshold);



		// Simpler version, without reusing
		// partial scores information...
		//----------------------------------
		unsigned int p1 = 0, p2 = 0;
		unsigned short score = 0;



		while (p1 < query_len && p2 < source_len)
		{
			unsigned int tkn1 = tokens[pos[query]+p1];
			unsigned int tkn2 = tokens[pos[source]+p2];

			if (
				(p1 == query_len-1 && tkn1 < tkn2) ||
				(p2 == source_len-1 && tkn2 < tkn1)
			)
				break;

			if (tkn1 == tkn2)
			{
				score++;
				p1++; p2++;
			}
			else
			{
				// Sophisticated solution, with positional filtering
				//---------------------------------------------------
				unsigned int whichset = tkn1 < tkn2 ? 1 : 2;
				unsigned int rem;
				if (whichset == 1) rem = (query_len  - p1) - 0;
				else               rem = (source_len - p2) - 0;

				if ((rem + score) < minoverlap) 
				{
					score = 0;
					break;
				}

				if (whichset == 1) p1++;
				else               p2++;

			} // if (tkn1 == tkn2)
		} // while (p1 < query_len && p2 < source_len)



		// Just testing minimum overlap score
		//------------------------------------
		float fscore = 0.0f;
		fscore += score;
		if (fscore >= minoverlap)
			scores[i] = score;
		else
			scores[i] = 0;

	} // for (i = 0; i < csize; i++)
} // void checking_kernel



/* DOCUMENTATION
 *
 */
__host__
double checking_cuda 
(
	unsigned int* d_tokens,  unsigned int* d_pos,     unsigned int* d_len,
	unsigned int* d_buckets, short* d_partial_scores, int*          d_nres,
	float threshold,         unsigned int csize,      unsigned int  n_sets,
	unsigned int**   similar_pairs_out,
	unsigned short** scores_out,
	int*             num_pairs_out
)
{
	unsigned long t0, t1, t00, t01;

	fprintf(stderr, "=> CHECKING...\n");
	t00 = ut_get_time_in_microseconds();

	dim3 grid, block;
	get_grid_config(grid, block);

	// ALLOCATING GPU MEMORY ---------------------------------------------------
	double mem = (double) (csize*sizeof(unsigned int))/1024.0;
	fprintf(stderr, "\t. Allocating %g KB on device (scores).\n", mem);

	unsigned short* d_scores;
	gpu(cudaMalloc(&d_scores, csize * sizeof(unsigned short)));
	gpu(cudaMemset(d_scores, 0, csize * sizeof(unsigned short)));
	//--------------------------------------------------------------------------

	// CALLING KERNEL ----------------------------------------------------------
	fprintf(stderr, "\t* Calling checking_kernel_cuda...\n");
	t0 = ut_get_time_in_microseconds();

	checking_kernel_cuda <<<grid,block>>> (
		d_buckets, d_scores, d_partial_scores,
		d_tokens, d_pos, d_len,
		threshold, n_sets, csize
	);
	gpu(cudaDeviceSynchronize());

	t1 = ut_get_time_in_microseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));
	//--------------------------------------------------------------------------

	// COMPACTING SIMILAR PAIRS ------------------------------------------------
	fprintf(stderr, "\t* Compacting similar pairs...\n");
	t0 = ut_get_time_in_microseconds();

	gpuAssert(cudaMemset(d_nres, 0, sizeof(int)));

	filter_k_2_cuda <<<grid,block>>>(
		d_scores, d_scores, d_buckets, d_buckets, d_nres, csize
	);
	gpu(cudaDeviceSynchronize());

	int nres;
	gpuAssert(cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));

	t1 = ut_get_time_in_microseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));
	//--------------------------------------------------------------------------

	unsigned int comp_pairs_size = (unsigned int) nres;


	// GETTING DATA BACK FROM GPU ----------------------------------------------
	fprintf(stderr, "\t* Getting data back from gpu memory...\n");
	t0 = ut_get_time_in_microseconds();

	unsigned short* scores;
	scores = (unsigned short*) malloc(csize * sizeof(unsigned short));
	gpu(cudaMemcpy(scores, d_scores, csize * sizeof(unsigned short), cudaMemcpyDeviceToHost));

	unsigned int* buckets;
	buckets = (unsigned int*) malloc(csize * sizeof(unsigned int));
	gpu(cudaMemcpy(buckets, d_buckets, csize * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	scores  = (unsigned short*) realloc(scores,  nres * sizeof(unsigned short));
	buckets = (unsigned int*)   realloc(buckets, nres * sizeof(unsigned int));

	t1 = ut_get_time_in_microseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", ut_interval_in_miliseconds(t0,t1));
	//--------------------------------------------------------------------------

	fprintf(stderr, "\t# NUMBER OF SIMILAR PAIRS: %u\n", comp_pairs_size);

	t01 = ut_get_time_in_microseconds();
	fprintf(stderr, "DONE IN %g ms\n\n", ut_interval_in_miliseconds(t00,t01));

	//------------------------------------
	*scores_out        = scores;
	*similar_pairs_out = buckets;
	*num_pairs_out     = comp_pairs_size;
	//------------------------------------

	return ut_interval_in_miliseconds(t00,t01);
}
