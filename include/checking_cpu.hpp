/* DOCUMENTATION

*/



#ifndef _CHECKING_H_
#define _CHECKING_H_



#include <vector>
#include "token.hpp"



using namespace std;



/* DOCUMENTATION
 *
 */
void checking_kernel (
	unsigned int* buckets, unsigned short* scores,
	vector< vector<token_t> > tsets, float threshold,
    short* partial_scores,
	unsigned int csize
);



/* DOCUMENTATION
 *
 */
double checking (
	vector< vector<token_t> > tsets, unsigned int* buckets,
	short* partial_scores, float threshold, unsigned int csize,
	unsigned int** similar_pairs, unsigned short** scores, int* num_pairs
);



/* DOCUMENTATION
 *
 */
void print_similar_pairs (
	vector< vector<token_t> > tsets, unsigned int* doc_index,
	unsigned int* similar_pairs, unsigned short* scores, int num_pairs
);



#endif // _CHECKING_H_
