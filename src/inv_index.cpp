#include "inv_index.hpp"
#include "similarity.hpp"



/* DOCUMENTATION
 *
 */
inv_index_t* idx_get_inv_index 
(
    vector< vector<token_t> >& tsets,
    unsigned long n_terms,
    unsigned long n_tokens,
    float threshold
)
{
    inv_index_t* inv_index = (inv_index_t*) malloc (sizeof(inv_index_t));
    unsigned int* pos = (unsigned int*) malloc (n_terms * sizeof(unsigned int));
    unsigned int* len = (unsigned int*) malloc (n_terms * sizeof(unsigned int));

    token_t** tkns  = (token_t**) malloc (n_tokens * sizeof(token_t*));
    token_t** lists = (token_t**) malloc (n_tokens * sizeof(token_t*));
    for (unsigned i = 0; i < n_tokens; i++) 
    {
        tkns[i] = (token_t*) malloc (sizeof(token_t));
        lists[i] = (token_t*) malloc (sizeof(token_t));
    }

    // Initializing vectors
    for (unsigned i = 0; i < n_terms; i++) 
    {
        pos[i] = 0;
        len[i] = 0;
    }

    int k = 0;
    for (unsigned i = 0; i < tsets.size(); i++) 
    {
        int setsize = tsets[i].size();
        int prefsize = h_jac_mid_prefix (setsize, threshold);
        for (int j = 0; j < prefsize; j++) 
        {
            tk_copy_token(tkns[k], &tsets[i][j]);
            k += 1;
        }
    }

    int tkns_size = k;

    // Counting token frequencies
    for (int i = 0; i < tkns_size; i++) 
    {
        len[tkns[i]->order_id] += 1;
    }

    // Performing prefix sum
    for (unsigned i = 1; i < n_terms; i++) 
    {
        pos[i] = pos[i-1] + len[i-1];
    }

    // Copying prefix sum to help build the index
    unsigned int postmp[n_terms];
    for (unsigned i = 0; i < n_terms; i++)
    {
        postmp[i] = pos[i];
    }

    // Building index
    for (int i = 0; i < tkns_size; i++)
    {
        int tid = tkns[i]->order_id;
        unsigned int idx = postmp[tid];

        tk_copy_token(lists[idx], tkns[i]);

        postmp[tid] += 1;
    }

    // Freeing tkns array
    for (int i = 0; i < tkns_size; i++)
    {
        free(tkns[i]);
    }
    free(tkns);

    // Realloc lists array
    for (unsigned i = tkns_size; i < n_tokens; i++)
    {
        free(lists[i]);
    }
    lists = (token_t**) realloc(lists, tkns_size * sizeof(token_t*));

    inv_index->pos = pos;
    inv_index->len = len;
    inv_index->lists = lists;
    inv_index->num_lists = n_terms;
    inv_index->num_tokens = n_tokens;
    inv_index->num_indexed_tokens = tkns_size;

    return inv_index;
}



/* DOCUMENTATION
 *
 */
entry_t* idx_convert_inv_index 
(
    inv_index_t* inv_index,
    unsigned int** pos_out, 
    unsigned int** len_out
)
{
    unsigned int n = inv_index->num_lists;
    unsigned int m = inv_index->num_indexed_tokens;

    unsigned int* pos = (unsigned int*) malloc (n * sizeof(unsigned int));
    unsigned int* len = (unsigned int*) malloc (n * sizeof(unsigned int));

    entry_t* lists = (entry_t*) malloc (m * sizeof(entry_t));

    for (unsigned i = 0; i < n; i ++) 
    {
        pos[i] = inv_index->pos[i];
        len[i] = inv_index->len[i];
    }

    for (unsigned i = 0; i < m; i++) 
    {
        lists[i].doc_id = inv_index->lists[i]->doc_id;
        lists[i].pos = inv_index->lists[i]->pos;
    }

    *pos_out = pos;
    *len_out = len;

    return lists;
}



/* DOCUMENTATION
 *
 */
void idx_print_inv_index 
(
    inv_index_t* inv_index
)
{
    printf("INVERTED INDEX LISTS:\n\n");

    for (unsigned i = 0; i < inv_index->num_lists; i++)
    {
        unsigned int pos = inv_index->pos[i];
        unsigned int len = inv_index->len[i];

        printf("initial position: %u\n", pos);
        printf("List for term id %d: ", i);

        for (unsigned j = 0; j < len; j++) 
        {
            printf("[%d] ", inv_index->lists[pos+j]->doc_id);
        }

        printf("\n\n");
    }
}
