#ifndef _FILTERING_H_
#define _FILTERING_H_

#include "inv_index.h"



void filtering_kernel_1 (
    inv_index_t* inv_index,
    vector< vector<token_t> > tsets,
    vector< vector<token_t> > tsets_2,
    short* scores, float threshold,
    int num_columns
);

double filtering_1 (
    vector< vector<token_t> > tsets,
    vector< vector<token_t> > tsets_2,
    inv_index_t* inv_index, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
);



//==============================================================================



void filtering_kernel_2 (
    inv_index_t* inv_index,
    vector< vector<token_t> > tsets,
    short* scores, float threshold
);

double filtering_2 (
    vector< vector<token_t> > tsets,
    inv_index_t* inv_index, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
);



//==============================================================================



void filtering_kernel_3 (
    inv_index_t* inv_index_1, inv_index_t* inv_index_2,
    vector< vector<token_t> > tsets_1, vector< vector<token_t> > tsets_2,
    short* scores, float threshold
);

double filtering_3 (
    vector< vector<token_t> > tsets_1, vector< vector<token_t> > tsets_2,
    inv_index_t* inv_index_1, inv_index_t* inv_index_2, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
);



//==============================================================================



void filtering_4_1 (
    vector< vector<token_t> > tsets, inv_index_t* inv_index, float threshold,
    short** partial_scores_out
);

void filtering_4_2 (
    vector< vector<token_t> > tsets, inv_index_t* inv_index, float threshold,
    short** partial_scores_out
);

void merge_4 (
    short*  partial_scores_1,
    short*  partial_scores_2,
    short* partial_scores_out, int size
);

double filtering_4 (
    vector< vector<token_t> > tsets_1, vector< vector<token_t> > tsets_2,
    inv_index_t* inv_index_1, inv_index_t* inv_index_2, float threshold,
    unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
);



#endif // _FILTERING_H_
