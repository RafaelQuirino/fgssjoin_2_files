/* DOCUMENTATION

*/



#ifndef _FILTERING_H_
#define _FILTERING_H_

#include "inv_index_cpu.hpp"

using namespace std;



/* DOCUMENTATION
 *
 */
void filtering_kernel 
(
    inv_index_t* inv_index,
    vector< vector<token_t> > tsets,
    short* scores, float threshold
);



/* DOCUMENTATION
 *
 */
double filtering 
(
    // Input arguments
    vector< vector<token_t> > tsets,
    inv_index_t* inv_index, float threshold,

    // Output arguments
	unsigned int** candidates, unsigned int* candidates_size,
    short** partial_scores_out
);



#endif // _FILTERING_H_
