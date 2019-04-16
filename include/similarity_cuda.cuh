#ifndef _SIMILARITY_CUDA_CUH_
#define _SIMILARITY_CUDA_CUH_



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
float jac_min_size (unsigned int size, float threshold)
{
	return (float) size * threshold;
}



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
float jac_max_size (unsigned int size, float threshold)
{
	return (float) size / threshold;
}



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
float jac_min_overlap (unsigned int size1, unsigned int size2, float threshold)
{
	float t = threshold;
	return (float) (t/(1 + t)) * ((float) (size1 + size2));
}



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
unsigned jac_max_prefix (unsigned int set_size, float threshold)
{
	return set_size - ceil(jac_min_size(set_size, threshold)) + 1;
}



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
unsigned jac_mid_prefix (unsigned int set_size, float threshold)
{
	return set_size - ceil(jac_min_overlap(set_size, set_size, threshold)) + 1;
}



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
float jac_similarity (unsigned int size1, unsigned int size2, unsigned int inter)
{
	unsigned int uni = size1 + size2 - inter;
	return (float) inter/ (float) uni;
}



/* DOCUMENTATION
 *
 */
__host__ __device__ extern inline
float jac_similarity (unsigned int size1, unsigned int size2, unsigned short inter)
{
	unsigned int uni = size1 + size2 - (unsigned int) inter;
	return (float) inter/ (float) uni;
}



#endif // _SIMILARITY_CUDA_CUH_
