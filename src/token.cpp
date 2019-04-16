#include <string.h>  // strlen
#include <algorithm> // sort

#include "util.hpp"
#include "sort.hpp"
#include "qgram.hpp"
#include "token.hpp"



// Comparing functions (for sorting) ------------------------------------------
bool tk_compare_freq (token_t t1, token_t t2)
{
	return t1.freq < t2.freq;
}

bool tk_compare_qgram (token_t t1, token_t t2)
{
	return strcmp (t1.qgram.c_str(), t2.qgram.c_str()) < 0;
}

bool compare_order (token_t a, token_t b)
{
	return a.order_id < b.order_id;
}
//-----------------------------------------------------------------------------



/*
 * Build the token sets using the records and the dictionary. 
 * Returns the token sets.
 */
vector< vector<token_t> >
tk_get_tokensets (
	unordered_map<unsigned long,token_t> dict,
	vector< vector<string> > records
)
{
    vector< vector<token_t> > tsets;
	unsigned long t0, t1;



	fprintf (stderr, "\tBuilding token sets...\n");
    t0 = ut_get_time_in_microseconds();

    for (unsigned int i = 0; i < records.size(); i++)
    {
		vector<string> record = records[i];
        vector<token_t> set;

        for (unsigned int j = 0; j < record.size(); j++)
        {
            string qgram = record[j];
			unsigned long hcode = qg_hash(qgram);

            unordered_map<unsigned long,token_t>::const_iterator result;
			result = dict.find(hcode);

            if (result == dict.end()) {
                fprintf(stderr, "Error in tk_get_tokensets, line %d. Token not in dictionary.\n", __LINE__);
            }
            else {
                token_t newtkn;
				newtkn.qgram = qgram;
				newtkn.hash = dict[hcode].hash;
                newtkn.freq = dict[hcode].freq;
                newtkn.doc_id   = i;
                newtkn.order_id = -1;
				newtkn.pos      = -1;

                set.push_back(newtkn);
            }

        } // for (int j = 0; j < recs[i].size(); j++)

        tsets.push_back(set);

    } // for (int i = 0; i < recs.size(); i++)

	t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done. It took %gms.\n", ut_interval_in_miliseconds (t0, t1));



	fprintf (stderr, "\tAssigning order ids (sorting tokens in corpus)...\n");
    t0 = ut_get_time_in_microseconds();

	tk_set_orders (tsets,dict);

	t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done. It took %gms.\n", ut_interval_in_miliseconds (t0, t1));

    return tsets;
}



/* DOCUMENTATION
 * 
 */
unsigned int* tk_convert_tokensets (
	vector< vector<token_t> > tsets, int num_tokens,
	unsigned int** pos_out, unsigned int** len_out
)
{
	unsigned int n = tsets.size();
	unsigned int* pos = (unsigned int*) malloc(n * sizeof(unsigned int));
	unsigned int* len = (unsigned int*) malloc(n * sizeof(unsigned int));
	unsigned int* lists = (unsigned int*) malloc(num_tokens * sizeof(unsigned int));

	for (unsigned int i = 0; i < n; i++)
		len[i] = tsets[i].size();

	pos[0] = 0;
	for (unsigned int i = 1; i < n; i++)
		pos[i] = pos[i-1] + len[i-1];

	int k = 0;
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < tsets[i].size(); j++) {
			lists[k] = tsets[i][j].order_id;
			k += 1;
		}
	}

	*pos_out = pos;
	*len_out = len;

	return lists;
}



/* DOCUMENTATION
 * 
 */
void tk_copy_token (token_t* dst, token_t* src)
{
	dst->hash = src->hash;
	dst->freq = src->freq;
	dst->doc_id = src->doc_id;
	dst->order_id = src->order_id;
	dst->pos = src->pos;
}



/* DOCUMENTATION
 * 
 */
void tk_print_token (token_t tkn)
{
	printf("hash     : %lu\n", tkn.hash);
	printf("freq     : %u\n", tkn.freq);
	printf("doc_id   : %d\n", tkn.doc_id);
	printf("order_id : %d\n", tkn.order_id);
	printf("pos      : %d\n", tkn.pos);
	printf("------------------------------------------\n");
}



/* DOCUMENTATION
 * 
 */
void tk_print_tset (vector<token_t> tset, int field)
{
	printf("[");
	for (unsigned int i = 0; i < tset.size(); i++) {
		char c = i == tset.size()-1 ? ']' : ',';
		if (field == QGRAM)
			printf("%s%c", tset[i].qgram.c_str(), c);
		else if (field == HASH)
			printf("%lu%c", tset[i].hash, c);
		else if (field == FREQ)
			printf("%u%c", tset[i].freq, c);
		else if (field == DOC_ID)
			printf("%d%c", tset[i].doc_id, c);
		else if (field == ORDER_ID)
			printf("%d%c", tset[i].order_id, c);
		else if (field == POS)
			printf("%d%c", tset[i].pos, c);
	}
	printf("\n");
}



/* DOCUMENTATION
 * 
 */
void tk_print_tsets (vector< vector<token_t> > tsets, int field)
{
	for (unsigned int i = 0; i < tsets.size(); i++)
		tk_print_tset(tsets[i], field);
}



/* DOCUMENTATION
 * 
 */
void tk_sort_freq (
    unsigned int* freqs, int* idx_freqs, int setsize
)
{
    // Just aliases
    unsigned int* a = freqs;
    int*          b = idx_freqs;
    int           n = setsize;

    // Sorting algorithm
    // (enhanced insertion sort)
    unsigned int i, j, d, e, from = 0, to = n;
    for ( i = from+1; i < to; i++ ) {
        d = a[i];
        e = b[i];
        unsigned int left = from, right = i-1;
        if ( a[right] > d ) {
            while ( right - left >= 2 ) {
                unsigned int middle = (right-left) / 2 + left - 1;
                if ( a[middle] > d ) right = middle;
                else left = middle + 1;
            }
            if ( right-left == 1 ) {
                unsigned int middle = left;
                if ( a[middle] > d ) right = middle;
                else left = middle + 1;
            }
            for ( j = i; j > left; j-- ) {
                a[j] = a[j-1];
                b[j] = b[j-1];
            }
            a[j] = d;
            b[j] = e;
        }
    }
}



/* DOCUMENTATION
 * 
 */
void tk_get_orders (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long,token_t>& dict
)
{
	// Copying dict into an array to be sorted
	int k = 0;
	int dict_size = dict.size();
	vector<token_t> tkns; // Using vector to be able to use sort algorithm
	for (pair<unsigned long,token_t> element:dict) {
		tkns.push_back(element.second);
		k += 1;
	}

	// Sorting algorithm
	sort (tkns.begin(), tkns.end(), tk_compare_freq);

	// Setting order_id in dictionary
	for (int i = 0; i < dict_size; i++) {
		dict[tkns[i].hash].order_id = i;
	}

	// Enable for tests
	//------------------
	// qg_print_dict(dict);
    // printf("\n");

	// Setting the appropriate orders in the token sets
	for (unsigned int i = 0; i < tsets.size(); i++) {
		for (unsigned int j = 0; j < tsets[i].size(); j++) {
			token_t term = dict[tsets[i][j].hash];
			tsets[i][j].order_id = term.order_id;
		}
	}
}



/* DOCUMENTATION
 * 
 */
void tk_get_freq_pos (vector< vector<token_t> >& tsets)
{
	for (unsigned int x = 0; x < tsets.size(); x++)
	{
		int setsize = tsets[x].size();
		int ids[setsize], pos[setsize];
		for (int y = 0; y < setsize; y++)
		{
			ids[y] = tsets[x][y].order_id;
			pos[y] = y;
		}

		// Sorting algorithm
	    // (enhanced insertion sort)
		int d, e;
		unsigned int n = setsize;
	    unsigned int i, j, from = 0, to = n;
	    for ( i = from+1; i < to; i++ ) {
	        d = ids[i];
	        e = pos[i];
	        unsigned int left = from, right = i-1;
	        if ( ids[right] > d ) {
	            while ( right - left >= 2 ) {
	                unsigned int middle = (right-left) / 2 + left - 1;
	                if ( ids[middle] > d ) right = middle;
	                else left = middle + 1;
	            }
	            if ( right-left == 1 ) {
	                unsigned int middle = left;
	                if ( ids[middle] > d ) right = middle;
	                else left = middle + 1;
	            }
	            for ( j = i; j > left; j-- ) {
	                ids[j] = ids[j-1];
	                pos[j] = pos[j-1];
	            }
	            ids[j] = d;
	            pos[j] = e;
	        }
	    } // Sorting algorithm

		// Setting the position
		for (int y = 0; y < setsize; y++) {
			tsets[x][pos[y]].pos = y;
		}

	} //for (int x = 0; i < tsets.size(); x++)
}



/* DOCUMENTATION
 * 
 */
void tk_set_orders (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long, token_t>& dict
)
{
	tk_get_orders (tsets,dict);
	tk_get_freq_pos (tsets);
}



/*
 * Returns an index array to find original positions
 * of records before sorting
 */
unsigned int*
tk_sort_sets (vector< vector<token_t> >& tsets)
{
    unsigned int num_sets = tsets.size();

    unsigned int* index;
    index = (unsigned int*) malloc (num_sets * sizeof(unsigned int));
    for (unsigned int i = 0; i < num_sets; i++)
        index[i] = i;

	// Sorting algorithm
	sort_vec_idx_by_size (tsets, index, tsets.size());

    // Fix doc_id/record_id for all tokens now...
    for (unsigned i = 0; i < tsets.size(); i++) {
        for (unsigned j = 0; j < tsets[i].size(); j++) {
            tsets[i][j].doc_id = i;
        }
    }

    return index;
}



/* DOCUMENTATION
 *
 */
void tk_sort_freq (vector< vector<token_t> >& tsets)
{
    for (unsigned int i = 0; i < tsets.size(); i++)
        sort (tsets[i].begin(), tsets[i].end(), compare_order);
}
