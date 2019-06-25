#include "filtering.h"
#include "similarity.h"
#include "compaction.h"
#include "util.h"



void filtering_kernel_1 (
    inv_index_t* inv_index,
    vector< vector<token_t> > tsets,
    vector< vector<token_t> > tsets_2,
    short* scores, float threshold,
    int num_columns
)
{
    unsigned int n = tsets.size();

    // For each query set...
    for (unsigned int i = 0; i < n; i++)
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
            unsigned int list_size = inv_index->len[token.order_1_id];
            unsigned int list_index = inv_index->pos[token.order_1_id];

            // For each source token in the inv_index list of query token...
			for (unsigned k = 0; k < list_size; k++)
			{
                token_t* tkn = inv_index->lists[list_index + k];
                unsigned int source = (unsigned int) tkn->doc_id;
                unsigned int source_tpos = (unsigned int) tkn->pos_1;

				if (true)//(query < source)
                {
                    unsigned int source_len = tsets_2[source].size();
					if (source_len < query_jac_min_size ||
                        source_len > query_jac_max_size)
                            continue;

                    //unsigned long bucket = (query * n) + source;
                    unsigned long bucket = (query * num_columns) + source;
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
                } // if (true)//(query < set)
			} // for (unsigned k = 0; k < list_size; k++)
		} // for (unsigned j = 0; j < prefix_size; j++)
	} // for (int i = 0; i < tsets.size(); i++)
} // void filtering_kernel_1



double filtering_1 (
    vector< vector<token_t> > tsets,
    vector< vector<token_t> > tsets_2,
    inv_index_t* inv_index, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
)
{
    int n_sets = tsets.size();
    int n_sets_2 = tsets_2.size();
    unsigned long t0, t1, t00, t01;
    double mem;

    fprintf(stderr, "=> FILTERING, 1st version (frequency order)...\n");
    t00 = getTimeInMicroseconds();

    // CREATING SCORES ARRAY (WHICH WILL CONTAIN THE "BUCKETS") ----------------
	unsigned int bsize = n_sets * n_sets_2;
    mem = (double)(bsize*sizeof(short))/(1024.0 * 1024.0);
	fprintf(stderr, "\t. Allocating %gMB on cpu (partial scores).\n", mem);

	short* partial_scores;
    partial_scores = (short*) malloc(bsize * sizeof(short));
    memset(partial_scores, (short) 0, bsize * sizeof(short));
    //--------------------------------------------------------------------------

    // CALLING THE FILTERING KERNEL --------------------------------------------
    fprintf(stderr, "\t* Calling filtering_kernel, 1st version...\n");
	t0 = getTimeInMicroseconds();

	filtering_kernel_1 (
        inv_index, tsets, tsets_2,
        partial_scores, threshold,
        tsets_2.size()
	);

	t1 = getTimeInMicroseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", intervalInMiliseconds(t0,t1));
    //--------------------------------------------------------------------------

    // COMPACTING FILTERED CANDIDATES ------------------------------------------
	fprintf(stderr, "\t* Compacting filtered buckets...\n");
	t0 = getTimeInMicroseconds();

    int nres = 0;
    unsigned int *comp_buckets;
    mem = (double)(bsize*sizeof(int))/(1024.0*1024.0);
    fprintf(stderr, "\t. Allocating %g MB on cpu (compacted buckets).\n", mem);

    comp_buckets = (unsigned int*) malloc(bsize * sizeof(unsigned int));

    filter_k (comp_buckets, partial_scores, &nres, bsize);
    unsigned int comp_buckets_size = (unsigned int) nres;
    comp_buckets = (unsigned int*) realloc(comp_buckets, nres * sizeof(unsigned int));

	t1 = getTimeInMicroseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", intervalInMiliseconds(t0,t1));
    //--------------------------------------------------------------------------

	*candidates = comp_buckets;
    *candidates_size = comp_buckets_size;
    *partial_scores_out = partial_scores;

    fprintf(stderr, "\t# NUMBER OF CANDIDATE PAIRS: %d\n", nres);

    t01 = getTimeInMicroseconds();
    fprintf(stderr, "DONE IN %g ms.\n\n", intervalInMiliseconds(t00,t01));

    return intervalInMiliseconds(t00,t01);

} // void filtering_1



//==============================================================================


/*
void filtering_kernel_2 (
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
            unsigned int list_size = inv_index->len[token.order_2_id];
            unsigned int list_index = inv_index->pos[token.order_2_id];

            // For each source token in the inv_index list of query token...
			for (unsigned k = 0; k < list_size; k++)
			{
                token_t* tkn = inv_index->lists[list_index + k];
                unsigned int source = (unsigned int) tkn->doc_id;
                unsigned int source_tpos = (unsigned int) tkn->pos_2;

				if (query < source)
                {
                    unsigned int source_len = tsets[source].size();
					if (source_len < query_jac_min_size ||
                        source_len > query_jac_max_size)
                            continue;

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
} // void filtering_kernel_2



double filtering_2 (
    vector< vector<token_t> > tsets, inv_index_t* inv_index, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
)
{
    int n_sets = tsets.size();
    unsigned long t0, t1, t00, t01;

    cout << "=> FILTERING, 2st version. (Normal, order 2).\n";
    t00 = getTimeInMicroseconds();

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
    cout << "\t* Calling filtering_kernel, 2nd version...\n";
	t0 = getTimeInMicroseconds();

	filtering_kernel_2 (
        inv_index, tsets,
        partial_scores, threshold
	);

	t1 = getTimeInMicroseconds();
	cout << "\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

    // COMPACTING FILTERED CANDIDATES ------------------------------------------
	cout << "\t* Compacting filtered buckets...\n";
	t0 = getTimeInMicroseconds();

    int nres = 0;
    unsigned int *comp_buckets;

    cout << "\t. Allocating " << (double)(bsize*sizeof(int))/(1024*1024) <<
        " MB on cpu (compacted buckets).\n";

    comp_buckets = (unsigned int*) malloc(bsize * sizeof(unsigned int));

    filter_k (comp_buckets, partial_scores, &nres, bsize);
    unsigned int comp_buckets_size = (unsigned int) nres;
    comp_buckets = (unsigned int*) realloc(comp_buckets, nres * sizeof(unsigned int));

	t1 = getTimeInMicroseconds();
	cout << "\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

	*candidates = comp_buckets;
    *candidates_size = comp_buckets_size;
    *partial_scores_out = partial_scores;

	// cout << "\t\t. Freeing " << (double)(bsize*sizeof(short))/(1024*1024) <<
    //     " MB of device memory...\n";
    // free(partial_scores);

    cout << "\t# NUMBER OF CANDIDATE PAIRS: " << nres << "\n";

    t01 = getTimeInMicroseconds();
    cout << "DONE IN " << intervalInMiliseconds(t00,t01) << " ms.\n\n";

    return intervalInMiliseconds(t00,t01);

} // void filtering_2



//==============================================================================



void filtering_kernel_3 (
    inv_index_t* inv_index_1, inv_index_t* inv_index_2,
    vector< vector<token_t> > tsets_1, vector< vector<token_t> > tsets_2,
    short* scores, float threshold
)
{
    int n = tsets_1.size();

    // For each query set...
    for (int i = 0; i < n; i++)
	{
        unsigned int query = i;
        unsigned int query_len = tsets_1[i].size();

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

            token_t token = tsets_1[i][j];
            unsigned int list_size = inv_index_1->len[token.order_1_id];
            unsigned int list_index = inv_index_1->pos[token.order_1_id];

            // For each source token in the inv_index list of query token...
			for (unsigned k = 0; k < list_size; k++)
			{
                token_t* tkn = inv_index_1->lists[list_index + k];
                unsigned int source = (unsigned int) tkn->doc_id;
                unsigned int source_tpos = (unsigned int) tkn->pos_1;

				if (query < source)
                {
                    unsigned int source_len = tsets_1[source].size();
					if (source_len < query_jac_min_size ||
                        source_len > query_jac_max_size)
                            continue;

                    unsigned long bucket = (query * n) + source;
                    short score = scores[ bucket ];

                    if (score >= 0)
                    {
                        //------------------------------------------------------
                        // Now, test for at least one match in the prefixes of
                        // query and source records in the other order...
                        //------------------------------------------------------
                        int found_match = 0;
                        for (unsigned x = 0; x < query_jac_prefix_size; x++) {
                            token_t token2 = tsets_2[query][x];
                            unsigned int listsize2 = inv_index_2->len[token2.order_2_id];
                            unsigned int listindex2 = inv_index_2->pos[token2.order_2_id];
                            for (unsigned y = 0; y < listsize2; y++) {
                                token_t* tkn2 = inv_index_2->lists[listindex2 + y];
                                if (tkn2->doc_id == (int)source) {
                                    found_match = 1;
                                    break;
                                }
                            }
                            if (found_match == 1)
                                break;
                        }
                        if (found_match == 0) {
                            scores[ bucket ] = -1;
                        } //----------------------------------------------------
                        else {
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
                        }
                    } // if (score >= 0)
                } // if (query < set)
			} // for (unsigned k = 0; k < list_size; k++)
		} // for (unsigned j = 0; j < prefix_size; j++)
	} // for (int i = 0; i < tsets_1.size(); i++)
}



double filtering_3 (
    vector< vector<token_t> > tsets_1, vector< vector<token_t> > tsets_2,
    inv_index_t* inv_index_1, inv_index_t* inv_index_2, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
)
{
    int n_sets = tsets_1.size();
    unsigned long t0, t1, t00, t01;

    cout << "=> FILTERING, 3rd version. (Hybrid).\n";
    t00 = getTimeInMicroseconds();

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
    cout << "\t* Calling filtering_kernel, 3rd version...\n";
	t0 = getTimeInMicroseconds();

	filtering_kernel_3 (
        inv_index_1, inv_index_2, tsets_1, tsets_2,
        partial_scores, threshold
	);

	t1 = getTimeInMicroseconds();
	cout << "\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

    // COMPACTING FILTERED CANDIDATES ------------------------------------------
	cout << "\t* Compacting filtered buckets...\n";
	t0 = getTimeInMicroseconds();

    int nres = 0;
    unsigned int *comp_buckets;

    cout << "\t. Allocating " << (double)(bsize*sizeof(int))/(1024*1024) <<
        " MB on cpu (compacted buckets).\n";

    comp_buckets = (unsigned int*) malloc(bsize * sizeof(unsigned int));

    filter_k (comp_buckets, partial_scores, &nres, bsize);
    unsigned int comp_buckets_size = (unsigned int) nres;
    comp_buckets = (unsigned int*) realloc(comp_buckets, nres * sizeof(unsigned int));

	t1 = getTimeInMicroseconds();
	cout << "\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

	*candidates = comp_buckets;
    *candidates_size = comp_buckets_size;
    *partial_scores_out = partial_scores;

	// cout << "\t\t. Freeing " << (double)(bsize*sizeof(short))/(1024*1024) <<
    //     " MB of device memory...\n";
    // free(partial_scores);

    cout << "\t# NUMBER OF CANDIDATE PAIRS: " << nres << "\n";

    t01 = getTimeInMicroseconds();
    cout << "DONE IN " << intervalInMiliseconds(t00,t01) << " ms.\n\n";

    return intervalInMiliseconds(t00,t01);
}



//==============================================================================



void filtering_4_1 (
    vector< vector<token_t> > tsets, inv_index_t* inv_index, float threshold,
    short** partial_scores_out
)
{
    int n_sets = tsets.size();
    unsigned long t0, t1, t00, t01;

    cout << "\t-> FILTERING, 4_1th version...\n";
    t00 = getTimeInMicroseconds();

    // CREATING SCORES ARRAY (WHICH WILL CONTAIN THE "BUCKETS") ----------------
	unsigned int bsize = n_sets * n_sets;
    // cout << "\t\t- Potential candidates: " << ((bsize-n_sets)/2) << ".\n";
	cout << "\t\t. Allocating " << (double)(bsize*sizeof(short))/(1024 * 1024) <<
	   " MB on cpu (partial_scores_1).\n";

	short* partial_scores;
    partial_scores = (short*) malloc(bsize * sizeof(short));
    memset(partial_scores, (short) 0, bsize * sizeof(short));
    //--------------------------------------------------------------------------

    // CALLING THE FILTERING KERNEL --------------------------------------------
    cout << "\t\t* Calling filtering_kernel, 1st version...\n";
	t0 = getTimeInMicroseconds();

	filtering_kernel_1 (
        inv_index, tsets,
        partial_scores, threshold
	);

	t1 = getTimeInMicroseconds();
	cout << "\t\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

    // Outputing partial_scores table
    *partial_scores_out = partial_scores;

    t01 = getTimeInMicroseconds();
    cout << "\tDONE IN " << intervalInMiliseconds(t00,t01) << " ms.\n\n";

} // void filtering_4_1



void filtering_4_2 (
    vector< vector<token_t> > tsets, inv_index_t* inv_index, float threshold,
    short** partial_scores_out
)
{
    int n_sets = tsets.size();
    unsigned long t0, t1, t00, t01;

    cout << "\t-> FILTERING, 4_2nd version...\n";
    t00 = getTimeInMicroseconds();

    // CREATING SCORES ARRAY (WHICH WILL CONTAIN THE "BUCKETS") ----------------
	unsigned int bsize = n_sets * n_sets;
    // cout << "\t\t- Potential candidates: " << ((bsize-n_sets)/2) << ".\n";
	cout << "\t\t. Allocating " << (double)(bsize*sizeof(short))/(1024 * 1024) <<
	   " MB on cpu (partial_scores_2).\n";

	short* partial_scores;
    partial_scores = (short*) malloc(bsize * sizeof(short));
    memset(partial_scores, (short) 0, bsize * sizeof(short));
    //--------------------------------------------------------------------------

    // CALLING THE FILTERING KERNEL --------------------------------------------
    cout << "\t\t* Calling filtering_kernel, 2nd version...\n";
	t0 = getTimeInMicroseconds();

	filtering_kernel_2 (
        inv_index, tsets,
        partial_scores, threshold
	);

	t1 = getTimeInMicroseconds();
	cout << "\t\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

    // Outputing partial_scores table
    *partial_scores_out = partial_scores;

    t01 = getTimeInMicroseconds();
    cout << "\tDONE IN " << intervalInMiliseconds(t00,t01) << " ms.\n\n";

} // void filtering_4_1



void merge_4 (
    short*  partial_scores_1,
    short*  partial_scores_2,
    short* partial_scores_out, int size
)
{
    unsigned long t0, t1;

    cout << "\t-> MERGING...\n";
    t0 = getTimeInMicroseconds();

    for (int i = 0; i < size; i++)
    {
        if (partial_scores_1[i] > 0 &&
            partial_scores_2[i] > 0)
        {
            partial_scores_out[i] = partial_scores_1[i];
        }
    }

    t1 = getTimeInMicroseconds();
    cout << "\tDONE IN " << intervalInMiliseconds(t0,t1) << " ms.\n\n";
}



double filtering_4 (
    vector< vector<token_t> > tsets_1, vector< vector<token_t> > tsets_2,
    inv_index_t* inv_index_1, inv_index_t* inv_index_2, float threshold,
    unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
)
{
    unsigned long t0, t1, t00, t01;

    cout << "=> FILTERING, 4th version. (Independent).\n\n";
    t0 = getTimeInMicroseconds();

    short* partial_scores_1;
    short* partial_scores_2;

    short* partial_scores;
    unsigned int n_sets = tsets_1.size();
    unsigned int bsize = n_sets * n_sets;
    partial_scores = (short*) malloc(bsize * sizeof(short));
    memset(partial_scores, (short) 0, bsize * sizeof(short));

    filtering_4_1(tsets_1, inv_index_1, threshold, &partial_scores_1);
    filtering_4_2(tsets_2, inv_index_2, threshold, &partial_scores_2);
    merge_4(partial_scores_1, partial_scores_2, partial_scores, bsize);

    // COMPACTING FILTERED CANDIDATES ------------------------------------------
    cout << "\t-> COMPACTING...\n";
    t00 = getTimeInMicroseconds();

    int nres = 0;
    unsigned int *comp_buckets;
    comp_buckets = (unsigned int*) malloc(bsize * sizeof(unsigned int));

    filter_k (comp_buckets, partial_scores, &nres, bsize);
    unsigned int comp_buckets_size = (unsigned int) nres;
    comp_buckets = (unsigned int*) realloc(comp_buckets, nres * sizeof(unsigned int));

    t01 = getTimeInMicroseconds();
    cout << "\tDONE IN " << intervalInMiliseconds(t00,t01) << " ms.\n\n";
    //--------------------------------------------------------------------------

    *candidates = comp_buckets;
    *candidates_size = comp_buckets_size;
    *partial_scores_out = partial_scores;

	// cout << "\t\t. Freeing " << (double)(bsize*sizeof(short))/(1024*1024) <<
    //     " MB of device memory...\n";
    // free(partial_scores);

    cout << "\t# NUMBER OF CANDIDATE PAIRS: " << nres << "\n\n";

    t1 = getTimeInMicroseconds();
    cout << "DONE IN " << intervalInMiliseconds(t0,t1) << " ms.\n\n";

    return intervalInMiliseconds(t0,t1);
}
*/
