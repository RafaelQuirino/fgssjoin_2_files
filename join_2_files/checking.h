#ifndef _CHECKING_H_
#define _CHECKING_H_

#include <vector>
#include "token.h"

using namespace std;

void checking_kernel (
	unsigned int* buckets, unsigned short* scores,
	vector< vector<token_t> > tsets,
	vector< vector<token_t> > tsets_2,
	float threshold,
    short* partial_scores,
	unsigned int csize,
	int num_columns
);

double checking (
	vector< vector<token_t> > tsets,
	vector< vector<token_t> > tsets_2,
	unsigned int* buckets,
	short* partial_scores, float threshold, unsigned int csize,
	unsigned int** similar_pairs, unsigned short** scores, int* num_pairs
);



void checking_kernel_2 (
	vector< vector<token_t> > tsets_orig, unsigned int* doc_index,
	unsigned int* buckets, unsigned short* scores,
	vector< vector<token_t> > tsets, float threshold,
    short* partial_scores,
	unsigned int csize
);

double checking_2 (
	vector< vector<token_t> > tsets_orig, unsigned int* doc_index,
	vector< vector<token_t> > tsets, unsigned int* buckets,
	short* partial_scores, float threshold, unsigned int csize,
	unsigned int** similar_pairs, unsigned short** scores, int* num_pairs
);



void print_similar_pairs (
	vector<string> data,
	vector< vector<token_t> > tsets, unsigned int* doc_index,
	vector< vector<token_t> > tsets_2, unsigned int* doc_index_2,
	unsigned int* similar_pairs, unsigned short* scores, int num_pairs
);

#endif // _CHECKING_H_
