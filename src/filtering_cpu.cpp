#include "util.hpp"
#include "compaction_cpu.h"
#include "filtering_cpu.hpp"
#include "similarity_cpu.hpp"



/* DOCUMENTATION
 *
 */
void filtering_kernel (
    inv_index_t* inv_index,
    vector< vector<token_t> > tsets,
    short* scores, float threshold
)
{
    int n = tsets.size();

    // For each query set...
    for (int i = 0; i < n; i++)
	{
        unsigned int query = i;
        unsigned int query_len = tsets[i].size();

        unsigned int query_jac_prefix_size =
            h_jac_max_prefix (query_len, threshold);
		unsigned int query_jac_min_size =
            (unsigned int) floor(h_jac_min_size(query_len, threshold));
        unsigned int query_jac_max_size =
            (unsigned int) floor(h_jac_max_size(query_len, threshold));

        // For each query token in the prefix...
		for (unsigned j = 0; j < query_jac_prefix_size; j++)
		{
			unsigned int query_tpos = j;

            token_t token = tsets[i][j];
            unsigned int list_size = inv_index->len[token.order_id];
            unsigned int list_index = inv_index->pos[token.order_id];

            // For each source token in the inv_index list of query token...
			for (unsigned k = 0; k < list_size; k++)
			{
                token_t* tkn = inv_index->lists[list_index + k];
                unsigned int source = (unsigned int) tkn->doc_id;
                unsigned int source_tpos = (unsigned int) tkn->pos;

				if (query < source)
                {
                    unsigned int source_len = tsets[source].size();
					if (source_len < query_jac_min_size ||
                        source_len > query_jac_max_size) {
                        
                        continue;
                    }

                    unsigned long bucket = (query * n) + source;
                    short score = scores[ bucket ];

                    if (score >= 0)
                    {
                        unsigned int query_rem = query_len - query_tpos;
                        unsigned int source_rem = source_len - source_tpos;
                        unsigned int min_rem =
                            query_rem < source_rem ? query_rem : source_rem;
                        float jac_minoverlap =
                            h_jac_min_overlap (query_len, source_len, threshold);

                        if ((score + 1 + min_rem) < jac_minoverlap)
                            scores[ bucket ] = -1;
                        else
                            scores[ bucket ] += 1;

                    } // if (score >= 0)
                } // if (query < set)
			} // for (unsigned k = 0; k < list_size; k++)
		} // for (unsigned j = 0; j < prefix_size; j++)
	} // for (int i = 0; i < tsets.size(); i++)
} // void filtering_kernel_1



/* DOCUMENTATION
 *
 */
double filtering (
    vector< vector<token_t> > tsets, inv_index_t* inv_index, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
)
{
    int n_sets = tsets.size();
    unsigned long t0, t1, t00, t01;

    cout << "=> FILTERING, 1st version. (Normal, order 1).\n";
    t00 = ut_get_time_in_microseconds();

    // CREATING SCORES ARRAY (WHICH WILL CONTAIN THE "BUCKETS") ----------------
	unsigned int bsize = n_sets * n_sets;
    // cout << "\t- Potential candidates: " << ((bsize-n_sets)/2) << ".\n";
	cout << "\t. Allocating " << (double)(bsize*sizeof(short))/(1024 * 1024) <<
	   " MB on cpu (partial scores).\n";

	short* partial_scores;
    partial_scores = (short*) malloc(bsize * sizeof(short));
    memset(partial_scores, (short) 0, bsize * sizeof(short));
    //--------------------------------------------------------------------------

    // CALLING THE FILTERING KERNEL --------------------------------------------
    cout << "\t* Calling filtering_kernel, 1st version...\n";
	t0 = ut_get_time_in_microseconds();

	filtering_kernel (
        inv_index, tsets,
        partial_scores, threshold
	);

	t1 = ut_get_time_in_microseconds();
	cout << "\t> Done. It took " << ut_interval_in_miliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

    // COMPACTING FILTERED CANDIDATES ------------------------------------------
	cout << "\t* Compacting filtered buckets...\n";
	t0 = ut_get_time_in_microseconds();

    int nres = 0;
    unsigned int *comp_buckets;

    cout << "\t. Allocating " << (double)(bsize*sizeof(int))/(1024*1024) <<
        " MB on cpu (compacted buckets).\n";

    comp_buckets = (unsigned int*) malloc(bsize * sizeof(unsigned int));

    filter_k (comp_buckets, partial_scores, &nres, bsize);
    unsigned int comp_buckets_size = (unsigned int) nres;
    comp_buckets = (unsigned int*) realloc(comp_buckets, nres * sizeof(unsigned int));

	t1 = ut_get_time_in_microseconds();
	cout << "\t> Done. It took " << ut_interval_in_miliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

	*candidates = comp_buckets;
    *candidates_size = comp_buckets_size;
    *partial_scores_out = partial_scores;

	// cout << "\t\t. Freeing " << (double)(bsize*sizeof(short))/(1024*1024) <<
    //     " MB of device memory...\n";
    // free(partial_scores);

    cout << "\t# NUMBER OF CANDIDATE PAIRS: " << nres << "\n";

    t01 = ut_get_time_in_microseconds();
    cout << "DONE IN " << ut_interval_in_miliseconds(t00,t01) << " ms.\n\n";

    return ut_interval_in_miliseconds(t00,t01);

} // void filtering_1
