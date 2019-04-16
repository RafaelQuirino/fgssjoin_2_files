/* DOCUMENTATION

*/



#ifndef _FILTERING_H_
#define _FILTERING_H_



#include "inv_index.hpp"



using namespace std;



/* DOCUMENTATION
 *
 */
void filtering_kernel (
    inv_index_t* inv_index,
    vector< vector<token_t> > tsets,
    short* scores, float threshold
);



/* DOCUMENTATION
 *
 */
double filtering (
    vector< vector<token_t> > tsets,
    inv_index_t* inv_index, float threshold,
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
);



#endif // _FILTERING_H_
