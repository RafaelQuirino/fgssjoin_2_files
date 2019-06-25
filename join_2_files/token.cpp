#include "token.h"
#include "qgram.h"
#include "sort.h"
#include "util.h"
#include <algorithm> //lexicographic_compare, sort
#include <string.h>  //strlen

bool tk_compare_freq (token_t t1, token_t t2)
{
	return t1.freq < t2.freq;
}

bool tk_compare_qgram (token_t t1, token_t t2)
{
	return strcmp(t1.qgram.c_str(), t2.qgram.c_str()) < 0;
}

vector< vector<token_t> >
tk_get_tokensets (
	unordered_map<unsigned long,token_t> dict,
	vector< vector<string> > recs
)
{
    vector< vector<token_t> > tsets;

    for (unsigned int i = 0; i < recs.size(); i++)
    {
        vector<token_t> set;

        for (unsigned int j = 0; j < recs[i].size(); j++)
        {
            string qgram = recs[i][j];
			unsigned long hcode = qg_hash(qgram);

            unordered_map<unsigned long,token_t>::const_iterator result;
			result = dict.find(hcode);

            if (result == dict.end()) {
                fprintf(stderr, "Error in tk_get_tokensets. Token not in dictionary.\n");
            }
            else {
                token_t newtkn;
				newtkn.qgram = qgram;
				newtkn.hash = dict[hcode].hash;
                newtkn.freq = dict[hcode].freq;
                newtkn.doc_id       = i;
                newtkn.order_1_id   = -1;
				newtkn.pos_1        = -1;
                newtkn.order_2_id   = -1;
				newtkn.pos_2        = -1;

                set.push_back(newtkn);
            }

        } // for (int j = 0; j < recs[i].size(); j++)

        tsets.push_back(set);

    } // for (int i = 0; i < recs.size(); i++)

	unsigned long t0, t1;

	fprintf(stderr, "\tSetting orders...\n");
    t0 = getTimeInMicroseconds();

	tk_set_orders(tsets,dict);

	t1 = getTimeInMicroseconds();
    fprintf(stderr, "\t> Done. It took %gms.\n", intervalInMiliseconds(t0,t1));

    return tsets;
}



unsigned int* tk_convert_tokensets (
	vector< vector<token_t> > tsets, int num_tokens, int order,
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
			if (order == ORDER_1) lists[k] = tsets[i][j].order_1_id;
			else                  lists[k] = tsets[i][j].order_2_id;
			k += 1;
		}
	}

	*pos_out = pos;
	*len_out = len;

	return lists;
}



void tk_copy_token (token_t* dst, token_t* src)
{
    // dst->qgram = src->qgram; // this doesn't work...

	dst->hash = src->hash;
	dst->freq = src->freq;
	dst->doc_id = src->doc_id;
	dst->order_1_id = src->order_1_id;
	dst->pos_1 = src->pos_1;
	dst->order_2_id = src->order_2_id;
	dst->pos_2 = src->pos_2;
}



void tk_print_token (token_t tkn)
{
	printf("hash      : %lu\n", tkn.hash);
	printf("freq      : %u\n", tkn.freq);
	printf("doc_id    : %d\n", tkn.doc_id);
	printf("order_1_id: %d\n", tkn.order_1_id);
	printf("pos_1     : %d\n", tkn.pos_1);
	printf("order_2_id: %d\n", tkn.order_2_id);
	printf("pos_2     : %d\n", tkn.pos_2);
	printf("------------------------------------------\n");
}



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
		else if (field == ORDER_1_ID)
			printf("%d%c", tset[i].order_1_id, c);
		else if (field == POS_1)
			printf("%d%c", tset[i].pos_1, c);
		else if (field == ORDER_2_ID)
			printf("%d%c", tset[i].order_2_id, c);
		else if (field == POS_2)
			printf("%d%c", tset[i].pos_2, c);
	}
	printf("\n");
}



void tk_print_tsets (vector< vector<token_t> > tsets, int field)
{
	for (unsigned int i = 0; i < tsets.size(); i++)
		tk_print_tset(tsets[i], field);
}



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
    for (i = from+1; i < to; i++) {
        d = a[i];
        e = b[i];
        unsigned int left = from, right = i-1;
        if ( a[right] > d ) {
            while ( right - left >= 2 ) {
                unsigned int middle = (right-left)/2 + left - 1;
                if ( a[middle] > d ) right = middle;
                else left = middle + 1;
            }
            if ( right-left == 1 ) {
                unsigned int middle = left;
                if ( a[middle] > d ) right = middle;
                else left = middle + 1;
            }
            for (j = i; j > left; j--) {
                a[j] = a[j-1];
                b[j] = b[j-1];
            }
            a[j] = d;
            b[j] = e;
        }
    }
}



// bool tk_lexic_less (const char* a, const char* b)
// {
//     return lexicographic_compare(a,a+strlen(a),b,b+strlen(b));
// }

void tk_sort_lexic (
    const char* qgrams[], int* idx_qgrams, int setsize
)
{
    // Just aliases
    const char** a = qgrams;
    int*         b = idx_qgrams;
    int          n = setsize;

    // Sorting algorithm
    // (enhanced insertion sort)
    unsigned int i, j, e, from = 0, to = n;
    const char* d;
    for (i = from+1; i < to; i++) {
        d = a[i];
        e = b[i];
        unsigned int left = from, right = i-1;
        // if ( a[right] > d ) {
        if ( strcmp(a[right],d) > 0 ) {
            while ( right - left >= 2 ) {
                unsigned int middle = (right-left)/2 + left - 1;
                // if ( a[middle] > d ) right = middle;
                if ( strcmp(a[middle],d) > 0 ) right = middle;
                else left = middle + 1;
            }
            if ( right-left == 1 ) {
                unsigned int middle = left;
                // if ( a[middle] > d ) right = middle;
                if ( strcmp(a[middle],d) > 0 ) right = middle;
                else left = middle + 1;
            }
            for (j = i; j > left; j--) {
                a[j] = a[j-1];
                b[j] = b[j-1];
            }
            a[j] = d;
            b[j] = e;
        }
    }
}



//=========================================================================
void tk_get_orders (vector< vector<token_t> >& tsets)
{

    for (unsigned int i = 0; i < tsets.size(); i++)
    {
        //vector<token_t> set = tsets[i];
        int setsize = tsets[i].size();

        unsigned int freqs[setsize];
        const char*  qgrams[setsize];

        for (int j = 0; j < setsize; j++)
        {
            freqs[j]  = tsets[i][j].freq;
            qgrams[j] = tsets[i][j].qgram.c_str();
        }

        // Index vectors, to be able to assign the order_x_id
        // to the respective tokens in the token sets
        int idx_freqs[setsize];
        int idx_qgrams[setsize];
        for (int k = 0; k < setsize; k++) {
            idx_freqs[k]  = k;
            idx_qgrams[k] = k;
        }

        // First order: freq ------------------------------
        tk_sort_freq(freqs, idx_freqs, setsize);
        for (int k = 0; k < setsize; k++) {
            tsets[i][idx_freqs[k]].order_1_id = k;
        }
        //-------------------------------------------------

        // Second order: lexic (qgram) --------------------
        tk_sort_lexic(qgrams, idx_qgrams, setsize);
        for (int k = 0; k < setsize; k++) {
            tsets[i][idx_qgrams[k]].order_2_id = k;
        }
        //-------------------------------------------------
    }
}



void tk_get_orders_2 (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long,token_t>& dict
)
{
	// Copying dict into an array to be sorted
	int k = 0;
	int dict_size = dict.size();
	// token_t tkns[dict_size];
	// for (pair<unsigned long,token_t> element : dict) {
	// 	tk_copy_token(&tkns[k], &(element.second));
	// 	tkns[k].qgram = element.second.qgram;
	// 	k += 1;
	// }
	vector<token_t> tkns; // Using vector to be able to use sort algorithm
	for (pair<unsigned long,token_t> element:dict) {
		tkns.push_back(element.second);
		k += 1;
	}

	//---------------------------------------------------------------
	// FIRST ORDER: TOKEN FREQUENCY
	//---------------------------------------------------------------
	// Sorting algorithm
    // (enhanced insertion sort)
	// token_t d;
	// int n = dict_size;
	// unsigned int i, j, from = 0, to = n;
    // for (i = from+1; i < to; i++) {
    //     d = tkns[i];
    //     unsigned int left = from, right = i-1;
    //     if ( tkns[right].freq > d.freq ) {
    //         while ( right - left >= 2 ) {
    //             unsigned int middle = (right-left)/2 + left - 1;
    //             if ( tkns[middle].freq > d.freq ) right = middle;
    //             else left = middle + 1;
    //         }
    //         if ( right-left == 1 ) {
    //             unsigned int middle = left;
    //             if ( tkns[middle].freq > d.freq ) right = middle;
    //             else left = middle + 1;
    //         }
    //         for (j = i; j > left; j--) tkns[j] = tkns[j-1];
    //         tkns[j] = d;
    //     }
    // }

	// Better sorting algorithm...
	sort(tkns.begin(), tkns.end(), tk_compare_freq);

	// Setting order_1_id in dictionary
	for (int i = 0; i < dict_size; i++) {
		dict[tkns[i].hash].order_1_id = i;
	}
	//---------------------------------------------------------------

	//---------------------------------------------------------------
	// SECOND ORDER: QGRAM LEXICOGRAPHIC
	//---------------------------------------------------------------
	// Sorting algorithm
    // (enhanced insertion sort)
	// from = 0; to = n;
    // for (i = from+1; i < to; i++) {
    //     d = tkns[i];
    //     unsigned int left = from, right = i-1;
	// 	//if ( tkns[right].freq > d.freq ) {
	// 	if ( strcmp(tkns[right].qgram.c_str(), d.qgram.c_str()) >= 0 ) {
    //         while ( right - left >= 2 ) {
    //             unsigned int middle = (right-left)/2 + left - 1;
	// 			//if ( tkns[middle].freq > d.freq )
	// 			if ( strcmp(tkns[middle].qgram.c_str(), d.qgram.c_str()) >= 0 )
	// 				right = middle;
    //             else left = middle + 1;
    //         }
    //         if ( right-left == 1 ) {
    //             unsigned int middle = left;
	// 			//if ( tkns[middle].freq > d.freq )
	// 			if ( strcmp(tkns[middle].qgram.c_str(), d.qgram.c_str()) >= 0 )
	// 				right = middle;
    //             else left = middle + 1;
    //         }
    //         for (j = i; j > left; j--) tkns[j] = tkns[j-1];
    //         tkns[j] = d;
    //     }
    // }

	// Better sorting algorithm...
	sort(tkns.begin(), tkns.end(), tk_compare_qgram);

	// Setting order_2_id in dictionary
	for (int i = 0; i < dict_size; i++) {
		dict[tkns[i].hash].order_2_id = i;
	}
	//---------------------------------------------------------------

	// qg_print_dict(dict);
    // printf("\n");

	// Setting the appropriate orders in the token sets
	for (unsigned int i = 0; i < tsets.size(); i++) {
		for (unsigned int j = 0; j < tsets[i].size(); j++) {
			token_t term = dict[tsets[i][j].hash];
			tsets[i][j].order_1_id = term.order_1_id;
			tsets[i][j].order_2_id = term.order_2_id;
		}
	}
}



void tk_get_freq_pos (vector< vector<token_t> >& tsets)
{
	for (unsigned int x = 0; x < tsets.size(); x++)
	{
		int setsize = tsets[x].size();
		int ids[setsize], pos[setsize];
		for (int y = 0; y < setsize; y++)
		{
			ids[y] = tsets[x][y].order_1_id;
			pos[y] = y;
		}

		// Sorting algorithm
	    // (enhanced insertion sort)
		int d, e;
		unsigned int n = setsize;
	    unsigned int i, j, from = 0, to = n;
	    for (i = from+1; i < to; i++) {
	        d = ids[i];
	        e = pos[i];
	        unsigned int left = from, right = i-1;
	        if ( ids[right] > d ) {
	            while ( right - left >= 2 ) {
	                unsigned int middle = (right-left)/2 + left - 1;
	                if ( ids[middle] > d ) right = middle;
	                else left = middle + 1;
	            }
	            if ( right-left == 1 ) {
	                unsigned int middle = left;
	                if ( ids[middle] > d ) right = middle;
	                else left = middle + 1;
	            }
	            for (j = i; j > left; j--) {
	                ids[j] = ids[j-1];
	                pos[j] = pos[j-1];
	            }
	            ids[j] = d;
	            pos[j] = e;
	        }
	    } // Sorting algorithm

		// Setting the position
		for (int y = 0; y < setsize; y++) {
			tsets[x][pos[y]].pos_1 = y;
		}

	} //for (int x = 0; i < tsets.size(); x++)
}



void tk_get_lexic_pos (vector< vector<token_t> >& tsets)
{
	for (unsigned int x = 0; x < tsets.size(); x++)
	{
		int setsize = tsets[x].size();
		int ids[setsize], pos[setsize];
		for (int y = 0; y < setsize; y++)
		{
			ids[y] = tsets[x][y].order_2_id;
			pos[y] = y;
		}

		// Sorting algorithm
	    // (enhanced insertion sort)
		int d, e;
		unsigned int n = setsize;
	    unsigned int i, j, from = 0, to = n;
	    for (i = from+1; i < to; i++) {
	        d = ids[i];
	        e = pos[i];
	        unsigned int left = from, right = i-1;
	        if ( ids[right] > d ) {
	            while ( right - left >= 2 ) {
	                unsigned int middle = (right-left)/2 + left - 1;
	                if ( ids[middle] > d ) right = middle;
	                else left = middle + 1;
	            }
	            if ( right-left == 1 ) {
	                unsigned int middle = left;
	                if ( ids[middle] > d ) right = middle;
	                else left = middle + 1;
	            }
	            for (j = i; j > left; j--) {
	                ids[j] = ids[j-1];
	                pos[j] = pos[j-1];
	            }
	            ids[j] = d;
	            pos[j] = e;
	        }
	    } // Sorting algorithm

		// Setting the position
		for (int y = 0; y < setsize; y++) {
			tsets[x][pos[y]].pos_2 = y;
		}

	} //for (int x = 0; i < tsets.size(); x++)
}



//-------------------------------------------------------------------------
void tk_set_orders (
	vector< vector<token_t> >& tsets,
	unordered_map<unsigned long, token_t>& dict
)
{
	tk_get_orders_2(tsets,dict);
	tk_get_freq_pos(tsets);
	tk_get_lexic_pos(tsets);
}
//-------------------------------------------------------------------------
//=========================================================================



unsigned int* //index array to find original positions
              //of records before sorting
tk_sort_sets (vector< vector<token_t> >& tsets)
{
    unsigned int num_sets = tsets.size();

    unsigned int* index;
    index = (unsigned int*) malloc(num_sets * sizeof(unsigned int));
    for (unsigned int i = 0; i < num_sets; i++)
        index[i] = i;

    // Just aliases
    // vector< vector<token_t> >& a = tsets;
    // unsigned int*              b = index;
    // unsigned int               n = num_sets;

    // Sorting algorithm
    // (enhanced insertion sort)
	/*
    vector<token_t> d;
    unsigned int i, j, e, from = 0, to = n;
    for (i = from+1; i < to; i++) {
        d = a[i];
        e = b[i];
        unsigned int left = from, right = i-1;
        // if ( a[right] > d ) {
        if ( a[right].size() < d.size() ) {
            while ( right - left >= 2 ) {
                unsigned int middle = (right-left)/2 + left - 1;
                // if ( a[middle] > d ) right = middle;
                if ( a[middle].size() < d.size() ) right = middle;
                else left = middle + 1;
            }
            if ( right-left == 1 ) {
                unsigned int middle = left;
                // if ( a[middle] > d ) right = middle;
                if ( a[middle].size() < d.size() ) right = middle;
                else left = middle + 1;
            }
            for (j = i; j > left; j--) {
                a[j] = a[j-1];
                b[j] = b[j-1];
            }
            a[j] = d;
            b[j] = e;
        }
    }
	*/

	// Better sorting algorithm...
	sort_vec_idx_by_size(tsets,index,tsets.size());

    // Fix doc_id/record_id for all tokens now...
    for (unsigned i = 0; i < tsets.size(); i++) {
        for (unsigned j = 0; j < tsets[i].size(); j++) {
            tsets[i][j].doc_id = i;
        }
    }

    return index;
}



bool compare_freq (token_t a, token_t b)
{
	//return a.freq < b.freq;
	return a.order_1_id < b.order_1_id;
}
void tk_sort_freq (vector< vector<token_t> >& tsets)
{
    for (unsigned int i = 0; i < tsets.size(); i++)
        sort(tsets[i].begin(), tsets[i].end(), compare_freq);
}



bool compare_lexic (token_t a, token_t b)
{
	//return strcmp(a.qgram.c_str(),b.qgram.c_str()) < 0;
	return a.order_2_id < b.order_2_id;
}
void tk_sort_lexic (vector< vector<token_t> >& tsets)
{
    for (unsigned int i = 0; i < tsets.size(); i++)
        sort(tsets[i].begin(), tsets[i].end(), compare_lexic);
}
