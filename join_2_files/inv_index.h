#ifndef _INV_INDEX_H_
#define _INV_INDEX_H_

#include "token.h"



struct _inv_index
{
    unsigned int* pos;
    unsigned int* len;

    //unsigned int* lists;
    token_t** lists;

    unsigned int num_lists;
    unsigned int num_tokens;
    unsigned int num_indexed_tokens;
};

typedef struct _inv_index inv_index_t;



struct _entry
{
    unsigned int doc_id;
    unsigned int pos;
};

typedef struct _entry entry_t;



// Inverted index for the first order (freq)
inv_index_t* idx_get_inv_index_1 (
    vector< vector<token_t> >& tsets,
    unsigned long n_terms,
    unsigned long n_tokens,
    float threshold
);

// Inverted index for the second order (lexic)
inv_index_t* idx_get_inv_index_2 (
    vector< vector<token_t> >& tsets,
    unsigned long n_terms,
    unsigned long n_tokens,
    float threshold
);

entry_t* idx_convert_inv_index (
    inv_index_t* inv_index, int order,
    unsigned int** pos_out, unsigned int** len_out
);

void idx_print_inv_index (inv_index_t* inv_index);

#endif // _INV_INDEX_H_
