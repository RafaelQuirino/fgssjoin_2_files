#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_map>

// Reading command line args
#include "arguments.h"

#include "io.h"
#include "util.h"
#include "data.h"
#include "qgram.h"
#include "token.h"
#include "inv_index.h"
#include "filtering.h"
#include "checking.h"

// Cuda files
// #include "util_cuda.cuh"
// #include "filtering_cuda.cuh"
// #include "checking_cuda.cuh"

using namespace std;

int main (int argc, char** argv)
{
    // ARGUMENTS ---------------------------------------------------------------
    // Arguments args = get_arguments(argc,argv);
    string file_path = string(argv[1]);//args.file_path;
    string file_path_2 = string(argv[2]);//args.file_path;
    int qgram_size = atoi(argv[3]);//args.qgram;
    // string output_path = ;//args.output_path;
    int input_size = 0;//args.input_size;
    float threshold = atof(argv[4]);//args.threshold;
    //--------------------------------------------------------------------------

    // MEASURING PERFORMANCE
	unsigned long t0, t1, t00, t01, t000, t001;
    double total_time = 0.0;

    // OPTIONS
    int print_cand  = 0;
    int print_pairs = 1;

    // VARIABLES
    unsigned long n_terms;
    unsigned long n_sets;
    unsigned long n_sets_2;
    unsigned long n_tokens;
    unsigned long n_tokens_2;
    unsigned int* doc_index;
    unsigned int* doc_index_2;
    // vector<string> raw_data;
    // vector<string> proc_data;
    vector<string> data;
    vector<string> data_2;
    vector< vector<string> > docs;
    vector< vector<string> > docs_2;
    unordered_map<unsigned long,token_t> dict;
    vector< vector<token_t> > tsets;
    vector< vector<token_t> > tsets_2;
    inv_index_t* inv_index;



    fprintf(stderr, "+----------------------+\n");
    fprintf(stderr, "| PRE-PROCESSING PHASE |\n");
    fprintf(stderr, "+----------------------+\n\n");

    t0 = getTimeInMicroseconds();


    // READING DATA AND CREATING RECORDS ---------------------------------------
    fprintf(stderr, "Reading data and creating records...\n");
    t00 = getTimeInMicroseconds();

    // READING INPUT DATA
    // raw_data = get_input_data(file_path, input_size);
    data = get_input_data(file_path, input_size);
    data_2 = get_input_data(file_path_2, input_size);

    // for (unsigned int i = 0; i < data.size(); i++) {
    //     fprintf(stderr, "\n\n %u => <<%s>>\n", i+1, data[i].c_str());
    //     fflush(stderr);
    // }
    // PROCESSING DATA
    // proc_data = get_proc_data(raw_data);
    proc_data(data);
    proc_data(data_2);

    // CREATING QGRAM RECORDS
    // docs = qg_get_records(proc_data, qgram_size);
    docs = qg_get_records(data, qgram_size);
    docs_2 = qg_get_records(data_2, qgram_size);

    t01 = getTimeInMicroseconds();
    fprintf(stderr, "> Done in %g ms.\n\n", intervalInMiliseconds(t00,t01));
    //--------------------------------------------------------------------------

    // CREATING TOKEN DICTIONARY -----------------------------------------------
    fprintf(stderr, "Creating token dictionary...\n");
    t00 = getTimeInMicroseconds();

    // dict = qg_get_dict(docs);
    dict = qg_get_dict_2(docs,docs_2);

    n_terms  = dict.size();
    n_sets   = docs.size();
    n_sets_2 = docs_2.size();
    n_tokens = 0;
    for (unsigned int i = 0; i < docs.size(); i++)
        n_tokens += docs[i].size();
    n_tokens_2 = 0;
    for (unsigned int i = 0; i < docs_2.size(); i++)
        n_tokens_2 += docs_2[i].size();

    t01 = getTimeInMicroseconds();
    fprintf(stderr, "> Done in %g ms.\n\n", intervalInMiliseconds(t00,t01));

    fprintf(stderr, "n_terms   : %lu\n", n_terms);
    fprintf(stderr, "n_sets    : %lu\n", n_sets);
    fprintf(stderr, "n_tokens  : %lu\n\n", n_tokens);
    fprintf(stderr, "n_sets_2  : %lu\n", n_sets_2);
    fprintf(stderr, "n_tokens_2: %lu\n\n", n_tokens_2);
    //--------------------------------------------------------------------------

    // CREATING TOKEN SETS -----------------------------------------------------
    fprintf(stderr, "Creating token sets...\n");
    t00 = getTimeInMicroseconds();

    tsets   = tk_get_tokensets(dict,docs);
    tsets_2 = tk_get_tokensets(dict,docs_2);

    t01 = getTimeInMicroseconds();
    fprintf(stderr, "> Done in %g ms.\n\n", intervalInMiliseconds(t00,t01));
    //--------------------------------------------------------------------------

    // SORTING  AND PREPARING TOKENSETS ----------------------------------------
    fprintf(stderr, "Sorting token sets...\n");
    t00 = getTimeInMicroseconds();

    // Sort sets in decreasing size order.
    fprintf(stderr, "\tSorting sets by size...\n");
    t000 = getTimeInMicroseconds();

    doc_index   = tk_sort_sets(tsets);
    doc_index_2 = tk_sort_sets(tsets_2);

    t001 = getTimeInMicroseconds();
    fprintf(stderr, "\t> Done. It took %g ms.\n",
        intervalInMiliseconds(t000,t001));

    fprintf(stderr, "\tSorting each set by token frequency...\n");
    t000 = getTimeInMicroseconds();

    tk_sort_freq(tsets);
    tk_sort_freq(tsets_2);

    t001 = getTimeInMicroseconds();
    fprintf(stderr, "\t> Done. It took %g ms.\n",
        intervalInMiliseconds(t000,t001));

    t01 = getTimeInMicroseconds();
    fprintf(stderr, "> Done in %g ms.\n\n", intervalInMiliseconds(t00,t01));

    // int field = ORDER_1_ID;
    // tk_print_tsets(tsets, field);
    // fprintf(stderr, "\n");
    // tk_print_tsets(tsets_2, field);
    //--------------------------------------------------------------------------

    t1 = getTimeInMicroseconds();
    fprintf(stderr, "Pre-processing phase took: %g ms.\n\n",
        (double)intervalInMiliseconds(t0,t1)/1000.0);



    fprintf(stderr, "+-----------------+\n");
    fprintf(stderr, "| ALGORITHM PHASE |\n");
    fprintf(stderr, "+-----------------+\n\n");

    total_time = 0.0;
    fprintf(stderr, "* FGSSJOIN IN CPU *\n\n");

    fprintf(stderr, "Building index...\n");

    t0 = getTimeInMicroseconds();
    // CREATING INVERTED INDEX FOR ORDER 1 -------------------------------------
    inv_index = idx_get_inv_index_1(tsets_2, n_terms, n_tokens_2, threshold);
    //--------------------------------------------------------------------------

    t1 = getTimeInMicroseconds();
    fprintf(stderr, "> Done in %g ms.\n\n", intervalInMiliseconds(t0,t1));
    total_time += intervalInMiliseconds(t0,t1);

    //==========================================================================
    // FGSSJOIN ON CPU
    //==========================================================================

    // FILTERING - GENERATE CANDIDATES FOR ORDER 1 -----------------------------
    unsigned int* cand;
    unsigned int cand_size;
    short* partial_scores;

    total_time += filtering_1(
        tsets, tsets_2, inv_index, threshold,
        &cand, &cand_size, &partial_scores
    );

    // Printing candidates
    //---------------------
    if (print_cand) {
        for (unsigned int i = 0; i < cand_size; i++) {
            int doc1 = cand[i] / tsets_2.size();
            int doc2 = cand[i] % tsets_2.size();
            printf("(%d,%d)\n", doc_index[doc1]+1, doc_index_2[doc2]+1);
        }
        printf("\n");
    }
    //--------------------------------------------------------------------------

    // CHECKING - VERIFYING CANDIDATES FROM ORDER 1 ----------------------------
    unsigned int* similar_pairs;
    unsigned short* scores;
    int num_pairs;

    total_time += checking(
        tsets, tsets_2, cand, partial_scores, threshold, cand_size,
        &similar_pairs, &scores, &num_pairs
    );

    if (print_pairs) {
        print_similar_pairs(
            data,
            tsets, doc_index,
            tsets_2, doc_index_2,
            similar_pairs, scores, num_pairs
        );
    }
    //--------------------------------------------------------------------------

    // FREEING MEMORY ----------------------------------------------------------
    if (cand_size > 1) free(cand);
    if (cand_size > 1) free(partial_scores);
    // if (num_pairs > 1) free(similar_pairs); // Compacted candidates,
                                               // i.e., cand...
    if (num_pairs > 1) free(scores);
    //--------------------------------------------------------------------------

    fprintf(stderr, "\nAlgorithm phase took: %gs.\n\n", total_time/1000.0);
    //==========================================================================

    return 0;

} // int main (int argc, char** argv)
