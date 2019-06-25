#ifndef _TOKEN_H_
#define _TOKEN_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

#define ORDER_1  0
#define ORDER_2  1

#define QGRAM       0
#define HASH        1
#define FREQ        2
#define DOC_ID      3
#define ORDER_1_ID  4
#define POS_1       5
#define ORDER_2_ID  6
#define POS_2       7

struct _token
{
	string qgram;       // Token qgram.
	unsigned long hash; // Qgram hash code.
	unsigned int freq;  // Frequency among records.

	int doc_id; // Except when in dictionary (came from no specific record).
	            // In this case, set to -1.

	int order_1_id; // Id (or overall position) in order #1 (freq).
	int pos_1;      // Position in the set by order 1.

	int order_2_id; // Id (or overall position) in order #2 (lexic).
	int pos_2;      // Position in the set by order 2.
};

typedef struct _token token_t;

vector< vector<token_t> >
tk_get_tokensets (
	unordered_map<unsigned long,token_t> dict,
	vector< vector<string> > recs
);

unsigned int* tk_convert_tokensets (
	vector< vector<token_t> > tsets, int num_tokens, int order,
	unsigned int** pos_out, unsigned int** len_out
);

void tk_copy_token (token_t* dst, token_t* src);
void tk_print_token (token_t tkn);
void tk_print_tset (vector<token_t> tset, int field);
void tk_print_tsets (vector< vector<token_t> > tsets, int field);

void tk_sort_freq (
    unsigned int* freqs, int* idx_freqs, int setsize
);

void tk_sort_lexic (
    const char* qgrams[], int* idx_qgrams, int setsize
);

//==============================================================================
void tk_get_orders (vector< vector<token_t> >& tsets);
void tk_get_orders_2 (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long,token_t>& dict
);
void tk_get_freq_pos  (vector< vector<token_t> >& tsets);
void tk_get_lexic_pos (vector< vector<token_t> >& tsets);
//------------------------------------------------------------------------------
void tk_set_orders (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long, token_t>& dict
);
//------------------------------------------------------------------------------
//==============================================================================

unsigned int* //index array to original positions
tk_sort_sets (vector< vector<token_t> >& tsets);

void tk_sort_freq (vector< vector<token_t> >& tsets);
void tk_sort_lexic (vector< vector<token_t> >& tsets);

#endif // _TOKEN_H_
