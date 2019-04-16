/* DOCUMENTATION

*/



#ifndef _INV_INDEX_H_
#define _INV_INDEX_H_



#include "token.hpp"



using namespace std;



/* DOCUMENTATION
 *
 */
typedef struct inv_index_t
{
    unsigned int* pos;
    unsigned int* len;

    token_t** lists;

    unsigned int num_lists;
    unsigned int num_tokens;
    unsigned int num_indexed_tokens;

} inv_index_t;



/* DOCUMENTATION
 *
 */
typedef struct entry_t
{
    unsigned int doc_id;
    unsigned int pos;

} entry_t;



/* DOCUMENTATION
 *
 */
inv_index_t* idx_get_inv_index (
    vector< vector<token_t> >& tsets,
    unsigned long n_terms,
    unsigned long n_tokens,
    float threshold
);



/* DOCUMENTATION
 *
 */
entry_t* idx_convert_inv_index (
    inv_index_t* inv_index,
    unsigned int** pos_out, 
    unsigned int** len_out
);



/* DOCUMENTATION
 *
 */
void idx_print_inv_index (inv_index_t* inv_index);



#endif // _INV_INDEX_H_
