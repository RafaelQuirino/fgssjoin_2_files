#ifndef _SIMILARITY_H_
#define _SIMILARITY_H_



#include <vector>
#include <string>
#include <iostream>
#include <math.h>



using namespace std;



/*
 *  DEFINITIONS
 *  ------------
 *
 *  - delta : overlap threshold.
*/



/* DOCUMENTATION
 *
 */
int overlap ();
int prefix_size (vector<string> record, int delta);
vector<string> prefix (vector<string> record, int delta);





/* DOCUMENTATION
 *
 */
extern inline
float h_jac_min_size (unsigned int size, float threshold)
{
	return (float) size * threshold;
}



/* DOCUMENTATION
 *
 */
extern inline
float h_jac_max_size (unsigned int size, float threshold)
{
	return (float) size / threshold;
}



/* DOCUMENTATION
 *
 */
extern inline
float h_jac_min_overlap (unsigned int size1, unsigned int size2, float threshold)
{
	float t = threshold;
	return (float) (t/(1 + t)) * ((float) (size1 + size2));
}



/* DOCUMENTATION
 *
 */
extern inline
unsigned h_jac_max_prefix (unsigned int set_size, float threshold)
{
	return set_size - ceil(h_jac_min_size(set_size, threshold)) + 1;
}



/* DOCUMENTATION
 *
 */
extern inline
unsigned h_jac_mid_prefix (unsigned int set_size, float threshold)
{
	return set_size - ceil(h_jac_min_overlap(set_size, set_size, threshold)) + 1;
}


/* DOCUMENTATION
 *
 */
extern inline
float h_jac_similarity (unsigned int size1, unsigned int size2, unsigned int inter)
{
	unsigned int uni = size1 + size2 - inter;
	return (float) inter/ (float) uni;
}


/* DOCUMENTATION
 *
 */
extern inline
float h_jac_similarity (unsigned int size1, unsigned int size2, unsigned short inter)
{
	unsigned int uni = size1 + size2 - (unsigned int) inter;
	return (float) inter/ (float) uni;
}





/* DOCUMENTATION
 *
 */
extern inline
float h_cos_min_size (unsigned int size, float threshold)
{
	float t = threshold;
	return (float) size * (t * t);
}



/* DOCUMENTATION
 *
 */
extern inline
float h_cos_max_size (unsigned int size, float threshold)
{
	float t = threshold;
	return (float) size / (t * t);
}



/* DOCUMENTATION
 *
 */
extern inline
float h_cos_min_overlap (unsigned int size1, unsigned int size2, float threshold)
{
	return (float) threshold * sqrtf (size1 * size2);
}



/* DOCUMENTATION
 *
 */
extern inline
unsigned h_cos_max_prefix (unsigned int set_size, float threshold)
{
	return set_size - ceil(h_cos_min_size(set_size, threshold)) + 1;
}



/* DOCUMENTATION
 *
 */
extern inline
unsigned h_cos_mid_prefix (unsigned int set_size, float threshold)
{
	return set_size - ceil(h_cos_min_overlap(set_size, set_size, threshold)) + 1;
}



/* DOCUMENTATION
 *
 */
extern inline
float h_cos_similarity (unsigned int size1, unsigned int size2, unsigned int inter)
{
	float sim = (float) inter / (float) sqrtf(size1 * size2);
	return sim;
}



#endif
