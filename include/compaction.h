/* DOCUMENTATION

*/



#ifdef __cplusplus
extern "C" {
#endif

#ifndef _COMPACTION_H
#define _COMPACTION_H



/* DOCUMENTATION
 *
 */
void filter_k (unsigned int *dst, const short *src, int *dstsize, int n);



/* DOCUMENTATION
 *
 */
void filter_k_2 (
    unsigned short *dst1, unsigned short *src1,
    unsigned int *dst2, unsigned int *src2,
    int *dstsize, int n
);



#endif

#ifdef __cplusplus
}
#endif
