#include "compaction_cpu.h"

void filter_k (unsigned int *dst, const short *src, int *dstsize, int n)
{
    int i;
    for (i = 0; i < n; i ++)
    {
        if (src[i] > 0) {
            dst[(*dstsize)] = i;
            (*dstsize) += 1;
        }
    }
}

void filter_k_2 (
    unsigned short *dst1, unsigned short *src1,
    unsigned int *dst2, unsigned int *src2,
    int *dstsize, int n
)
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (src1[i] > 0) {
            dst1[(*dstsize)] = src1[i];
            dst2[(*dstsize)] = src2[i];
            (*dstsize) += 1;
        }
    }
}
