#ifndef _TOKEN_H_
#define _TOKEN_H_



#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>



#define QGRAM     0
#define HASH      1
#define FREQ      2
#define DOC_ID    3
#define ORDER_ID  4
#define POS       5



using namespace std;



/* DOCUMENTATION
 * 
 */
typedef struct token_t
{
	string qgram;       // Token qgram.
	unsigned long hash; // Qgram hash code.
	unsigned int freq;  // Frequency among records.

	int doc_id;   // Except when in dictionary (came from no specific record).
	              // In this case, set to -1.

	int order_id; // Id in frequency order.
	int pos;      // Position in the set in frequency order.

} token_t;



/* DOCUMENTATION
 * 
 */ 
vector< vector<token_t> > // Token sets
tk_get_tokensets (
	unordered_map<unsigned long,token_t> dict,
	vector< vector<string> > recs
);



/*
 * Converts the tokensets data into sets of
 * one-dimensional arrays for use in the GPU
 */
unsigned int* tk_convert_tokensets (
	vector< vector<token_t> > tsets, int num_tokens,
	unsigned int** pos_out, unsigned int** len_out
);



/*
 * Auxiliar functions for common procedures and debugging
 */
void tk_copy_token (token_t* dst, token_t* src);
void tk_print_token (token_t tkn);
void tk_print_tset (vector<token_t> tset, int field);
void tk_print_tsets (vector< vector<token_t> > tsets, int field);


/* DOCUMENTATION
 * 
 */
void tk_sort_freq (
    unsigned int* freqs, int* idx_freqs, int setsize
);



/* DOCUMENTATION
 * 
 */
//-----------------------------------------------------------------------------
void tk_get_orders (vector< vector<token_t> >& tsets);
void tk_get_orders_2 (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long,token_t>& dict
);

void tk_get_freq_pos  (vector< vector<token_t> >& tsets);

void tk_set_orders (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long, token_t>& dict
);
//-----------------------------------------------------------------------------



/*
 * Sort the set of sets by set sizes.
 * Returns an index array to the sets original positions before sorting
 * (index to original sets positions)
 */
unsigned int*
tk_sort_sets (vector< vector<token_t> >& tsets);



/* DOCUMENTATION
 *  
 */
void tk_sort_freq (vector< vector<token_t> >& tsets);



#endif // _TOKEN_H_
