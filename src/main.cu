// C libs
#include <stdio.h>
#include <stdlib.h>

// C++ libs
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_map>

// Reading command line args
#include "arguments.hpp"

// Common libs
#include "io.hpp"
#include "util.hpp"
#include "data.hpp"
#include "qgram.hpp"
#include "token.hpp"
#include "inv_index.hpp"

// CPU libs
#include "filtering.hpp"
#include "checking.hpp"

// Cuda libs
#include "util_cuda.cuh"
#include "filtering_cuda.cuh"
#include "checking_cuda.cuh"



using namespace std;



int main (int argc, char** argv)
{
    // ARGUMENTS
    Arguments args     = get_arguments (argc,argv);
    string file_path   = args.file_path;
    string output_path = args.output_path;
    int qgram_size     = args.qgram;
    int input_size     = args.input_size;
    float threshold    = args.threshold;

    // MEASURING PERFORMANCE (in multiple leves)
    unsigned long t0, t1, t00, t01, t000, t001;
    double total_time;

    // OPTIONS
    int cpu_mode    = 0;
    int print_cand  = 0;
    int print_pairs = 1;

    // VARIABLES
    int i;
    unsigned long n_sets;
    unsigned long n_terms;
    unsigned long n_tokens;
    unsigned int* doc_index;
    vector<string> raw_data;
    vector<string> proc_data;
    vector< vector<string> > docs;
    unordered_map<unsigned long,token_t> dict;
    vector< vector<token_t> > tsets;
    inv_index_t* inv_index;
    unsigned int *pos, *len, *tkns;
    unsigned int *pos_idx, *len_idx;
    entry_t *lists;



    fprintf (stderr, "+----------------------+\n");
    fprintf (stderr, "| PRE-PROCESSING PHASE |\n");
    fprintf (stderr, "+----------------------+\n\n");

    t0 = ut_get_time_in_microseconds();



    // READING DATA AND CREATING RECORDS ---------------------------------------
    fprintf (stderr, "Reading data and creating records...\n");
    t00 = ut_get_time_in_microseconds();

    // READING INPUT DATA
    raw_data = dat_get_input_data (file_path, input_size);

    // PROCESSING DATA
    proc_data = dat_get_proc_data (raw_data);

    // CREATING QGRAM RECORDS
    docs = qg_get_records (proc_data, qgram_size);

    t01 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds (t00,t01));
    //--------------------------------------------------------------------------



    // CREATING TOKEN DICTIONARY -----------------------------------------------
    fprintf(stderr, "Creating token dictionary...\n");
    t00 = ut_get_time_in_microseconds();

    dict = qg_get_dict (docs);

    n_sets = docs.size();
    n_terms = dict.size();
    n_tokens = 0;
    for (int i = 0; i < docs.size(); i++)
    {
        n_tokens += docs[i].size();
    }

    t01 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds (t00, t01));

    fprintf (stderr, "n_sets  : %lu\n", n_sets);
    fprintf (stderr, "n_terms : %lu\n", n_terms);
    fprintf (stderr, "n_tokens: %lu\n\n", n_tokens);
    //--------------------------------------------------------------------------



    // CREATING TOKEN SETS -----------------------------------------------------
    fprintf (stderr, "Creating token sets...\n");
    t00 = ut_get_time_in_microseconds();

    tsets = tk_get_tokensets (dict, docs);

    t01 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds (t00, t01));
    //--------------------------------------------------------------------------


    
    // SORTING AND PREPARING TOKENSETS ----------------------------------------
    fprintf (stderr, "Preparing token sets...\n");
    t00 = ut_get_time_in_microseconds();

    // Sort sets in decreasing size order.
    fprintf(stderr, "\tSorting corpus by set by size...\n");
    t000 = ut_get_time_in_microseconds();

    doc_index = tk_sort_sets (tsets);

    t001 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done. It took %gms.\n",
        ut_interval_in_miliseconds (t000, t001)
    );

    fprintf (stderr, "\tSorting each set by term frequency...\n");
    t000 = ut_get_time_in_microseconds();

    tk_sort_freq (tsets);

    t001 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done. It took %gms.\n",
        ut_interval_in_miliseconds (t000, t001)
    );

    t01 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds (t00, t01));
    //--------------------------------------------------------------------------



    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "Pre-processing phase took %gms.\n\n",
        ut_interval_in_miliseconds (t0, t1)
    );

    //exit (0);



    fprintf (stderr, "\n\n");
    fprintf (stderr, "+-----------------+\n");
    fprintf (stderr, "| ALGORITHM PHASE |\n");
    fprintf (stderr, "+-----------------+\n\n");

    fprintf (stderr, "Building index...\n");

    t0 = ut_get_time_in_microseconds();

    // CREATING INVERTED INDEX -------------------------------------------------
    inv_index = idx_get_inv_index (tsets, n_terms, n_tokens, threshold);
    //--------------------------------------------------------------------------

    t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %g ms.\n\n", ut_interval_in_miliseconds (t0, t1));



if (cpu_mode) {

    ut_print_separator ("=", 80);
    fprintf (stderr, "\nSTARTING CPU SECTION\n\n");


    //==========================================================================
    // 1ST VERSION
    //==========================================================================

    ut_print_separator ("=", 80); printf ("\n");

    fprintf (stderr, "\n * FGSSJOIN IN CPU *\n\n");

    total_time = 0.0;

    // FILTERING - GENERATE CANDIDATES -----------------------------------------
    unsigned int* cand_1;
    unsigned int cand_size_1;
    short* partial_scores_1;
    
    total_time += filtering (
        tsets, inv_index, threshold,
        &cand_1, &cand_size_1, &partial_scores_1
    );

    // Printing candidates
    //---------------------
    if (print_cand) 
    {
        for (i = 0; i < cand_size_1; i++) 
        {
            int doc1 = cand_1[i] / tsets.size();
            int doc2 = cand_1[i] % tsets.size();
            printf ("(%d,%d)\n", doc_index[doc1]+1, doc_index[doc2]+1);
        }
        printf ("\n");
    }
    //--------------------------------------------------------------------------

    // CHECKING - VERIFYING CANDIDATES -----------------------------------------
    unsigned int* similar_pairs_1;
    unsigned short* scores_1;
    int num_pairs_1;

    total_time += checking (
        tsets, cand_1, partial_scores_1, threshold, cand_size_1,
        &similar_pairs_1, &scores_1, &num_pairs_1
    );

    if (print_pairs) {
        print_similar_pairs (
            tsets, doc_index, similar_pairs_1, scores_1, num_pairs_1
        );
    }
    //--------------------------------------------------------------------------

    // FREEING MEMORY ----------------------------------------------------------
    if (cand_size_1 > 1) free(cand_1);
    if (cand_size_1 > 1) free(partial_scores_1);
    // if (num_pairs_1 > 1) free(similar_pairs_1); // Compacted candidates...
    if (num_pairs_1 > 1) free(scores_1);
    //--------------------------------------------------------------------------

    fprintf (stderr, "\nTotal execution time: %gs.\n\n", total_time / 1000.0);

} // if (cpu_mode)



else {

    // VARIABLES
    int device = 0;
    entry_t      *d_lists;
    unsigned int *d_tkns, *d_pos, *d_len;
    unsigned int *d_pos_idx, *d_len_idx;
    unsigned int  num_lists, num_indexed_tokens;
    short        *d_partial_scores;
    unsigned int *d_comp_buckets;
    int          *d_nres;

    fprintf (stderr, "STARTING GPU SECTION\n");
    fprintf (stderr, "SETTING DEVICE %d\n\n", device);
    gpu (cudaSetDevice(device));
    gpu (cudaDeviceReset());

    // Preparing data for gpu --------------------------------------------------
    tkns  = tk_convert_tokensets (tsets, n_tokens, &pos, &len);
    lists = idx_convert_inv_index (inv_index, &pos_idx, &len_idx);
    //--------------------------------------------------------------------------

    // Sending data to gpu memory ----------------------------------------------
    unsigned long gpumem = 0;

    gpumem += 2 * n_sets * sizeof(unsigned int);
    gpumem += n_tokens * sizeof(unsigned int);
    gpumem += 2 * inv_index->num_lists * sizeof(unsigned int);
    gpumem += inv_index->num_indexed_tokens * sizeof(unsigned int);

    gpumem += 2 * n_sets * sizeof(unsigned int);
    gpumem += n_tokens * sizeof(unsigned int);
    gpumem += n_sets * n_sets * sizeof(short);

    gpumem += n_sets * n_sets * sizeof(short);
    gpumem += n_sets * n_sets * sizeof(unsigned int);

    fprintf( stderr, "Allocating and sending %gMB to gpu memory...\n",
        (double) gpumem / (1024.0 * 1024.0)
    );
    t0 = ut_get_time_in_microseconds();

    // Token sets
    gpu( cudaMalloc (&d_pos, n_sets * sizeof(unsigned int)));
    gpu( cudaMalloc (&d_len, n_sets * sizeof(unsigned int)));
    gpu( cudaMalloc (&d_tkns, n_tokens * sizeof(unsigned int)));
    gpu( cudaMemcpy (d_pos, pos, n_sets * sizeof(unsigned int),
        cudaMemcpyHostToDevice)
    );
    gpu( cudaMemcpy (d_len, len, n_sets * sizeof(unsigned int),
        cudaMemcpyHostToDevice)
    );
    gpu( cudaMemcpy (d_tkns, tkns, n_tokens * sizeof(unsigned int),
        cudaMemcpyHostToDevice)
    );

    // Inverted index
    num_lists = inv_index->num_lists;
    num_indexed_tokens = inv_index->num_indexed_tokens;
    gpu( cudaMalloc (&d_pos_idx, num_lists * sizeof(unsigned int)));
    gpu( cudaMalloc (&d_len_idx, num_lists * sizeof(unsigned int)));
    gpu( cudaMalloc (&d_lists, num_indexed_tokens * sizeof(entry_t)));
    gpu( cudaMemcpy (d_pos_idx, pos_idx, num_lists * sizeof(unsigned int),
        cudaMemcpyHostToDevice)
    );
    gpu( cudaMemcpy (d_len_idx, len_idx, num_lists * sizeof(unsigned int),
        cudaMemcpyHostToDevice)
    );
    gpu( cudaMemcpy (d_lists, lists, num_indexed_tokens * sizeof(entry_t),
        cudaMemcpyHostToDevice)
    );

    // Partial scores
    gpu( cudaMalloc (&d_partial_scores, (n_sets * n_sets) * sizeof(short)));

    // Compacted buckets
    gpu( cudaMalloc (&d_comp_buckets, (n_sets * n_sets) * sizeof(unsigned int)));

    // Compacted size
    gpu( cudaMalloc (&d_nres, sizeof(int)));

    t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %g ms.\n\n", ut_interval_in_miliseconds (t0, t1));
    //--------------------------------------------------------------------------



    // Computing execution time
    total_time = 0.0;

    // FILTERING - GENERATE CANDIDATES -----------------------------------------
    unsigned int cand_size_2;

    total_time += filtering_cuda (
        d_tkns, d_pos, d_len, d_lists, d_pos_idx, d_len_idx,
        threshold, n_sets, d_comp_buckets, d_partial_scores, d_nres,
        &cand_size_2
    );
    //--------------------------------------------------------------------------

    // CHECKING - VERIFYING CANDIDATES -----------------------------------------
    unsigned int* similar_pairs_2;
    unsigned short* scores_2;
    int num_pairs_2;

    total_time += checking_cuda (
        d_tkns, d_pos, d_len,
        d_comp_buckets, d_partial_scores, d_nres,
        threshold, cand_size_2, n_sets,
        &similar_pairs_2, &scores_2, &num_pairs_2
    );

    if (print_pairs) {
        print_similar_pairs (
            tsets, doc_index, similar_pairs_2, scores_2, num_pairs_2
        );
    }
    //--------------------------------------------------------------------------

    fprintf(stderr, "\nTotal execution time: %gs.\n\n", total_time/1000.0);

} // else of "if (cpu_mode)"



    return 0;

} // int main (int argc, char** argv)
