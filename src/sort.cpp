#include "sort.hpp"



/* DOCUMENTATION
 *
 */
int partition_1 (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    int l, int r
)
{
    int i, j; //, t;
    unsigned int pivot;
    pivot = vec[l].size();
    i = l; j = r+1;

    while (1)
    {
        // THIS WILL SORT IN INCREASING ORDER
        //------------------------------------
        //do ++i; while ( arr[i] <= pivot && i <= r );
        //do --j; while ( arr[j] > pivot );

        // THIS WILL SORT IN DECREASING ORDER
        //------------------------------------
        do ++i; while ( vec[i].size() >= pivot && i <= r );
        do --j; while ( vec[j].size() < pivot );

        if ( i >= j ) break;

        // swap(&arr[i], &arr[j]);
        vector<token_t> temp1 = vec[i];
        vec[i] = vec[j];
        vec[j] = temp1;

        // swap(&arr2[i], &arr2[j]);
        unsigned int temp2 = arr2[i];
        arr2[i] = arr2[j];
        arr2[j] = temp2;
    }

    // swap(&arr[l], &arr[j]);
    vector<token_t> temp1 = vec[l];
    vec[l] = vec[j];
    vec[j] = temp1;

    // swap(&arr2[l], &arr2[j]);
    unsigned int temp2 = arr2[l];
    arr2[l] = arr2[j];
    arr2[j] = temp2;

    return j;
}

void quicksort_aux_1 (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    int l, int r
)
{
    int j;

    if ( l < r )
    {
        // divide and conquer
        j = partition_1(vec, arr2, l, r);
        quicksort_aux_1(vec, arr2, l, j-1);
        quicksort_aux_1(vec, arr2, j+1, r);
    }
}

void quicksort_1 (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    unsigned int n
)
{
    quicksort_aux_1(vec, arr2, 0, n-1);
}

void sort_vec_idx_by_size (
    vector< vector<token_t> >& vec, unsigned int* arr2,
    unsigned int n
)
{
    quicksort_1(vec,arr2,n);
}



//==============================================================================



/* DOCUMENTATION
 *
 */
int partition_2 (token_t* arr,int l, int r)
{
    int i, j; //, t;
    unsigned int pivot;
    pivot = arr[l].freq;
    i = l; j = r+1;

    while (1)
    {
        // THIS WILL SORT IN INCREASING ORDER
        //------------------------------------
        do ++i; while ( arr[i].freq <= pivot && i <= r );
        do --j; while ( arr[j].freq > pivot );

        // THIS WILL SORT IN DECREASING ORDER
        //------------------------------------
        // do ++i; while ( arr[i].freq >= pivot && i <= r );
        // do --j; while ( arr[j].freq < pivot );

        if ( i >= j ) break;

        // swap(&arr[i], &arr[j]);
        token_t temp1 = arr[i];
        arr[i] = arr[j];
        arr[j] = temp1;
    }

    // swap(&arr[l], &arr[j]);
    token_t temp1 = arr[l];
    arr[l] = arr[j];
    arr[j] = temp1;

    return j;
}

void quicksort_aux_2 (token_t* arr, int l, int r)
{
    int j;

    if ( l < r )
    {
        // divide and conquer
        j = partition_2(arr, l, r);
        quicksort_aux_2(arr, l, j-1);
        quicksort_aux_2(arr, j+1, r);
    }
}

void quicksort_2 (token_t* arr, unsigned int n)
{
    quicksort_aux_2(arr, 0, n-1);
}

void sort_tokens_by_freq (token_t* arr, unsigned int n)
{
    quicksort_2(arr,n);
}



//==============================================================================



/* DOCUMENTATION
 *
 */
int partition_3 (
    char** qgrams, unsigned int* index,
    int l, int r
)
{
    int i, j; //, t;
    char* pivot;
    pivot = qgrams[l];
    i = l; j = r+1;

    while (1)
    {
        // THIS WILL SORT IN INCREASING ORDER
        //------------------------------------
        do ++i; while ( strcmp(qgrams[i], pivot) < 0 && i <= r );
        do --j; while ( strcmp(qgrams[j], pivot) >= 0 );

        // THIS WILL SORT IN DECREASING ORDER
        //------------------------------------
        // do ++i; while ( vec[i].size() >= pivot && i <= r );
        // do --j; while ( vec[j].size() < pivot );

        if ( i >= j ) break;

        // swap(&arr[i], &arr[j]);
        char* temp1 = qgrams[i];
        qgrams[i] = qgrams[j];
        qgrams[j] = temp1;

        // swap(&arr2[i], &arr2[j]);
        unsigned int temp2 = index[i];
        index[i] = index[j];
        index[j] = temp2;
    }

    // swap(&arr[l], &arr[j]);
    char* temp1 = qgrams[l];
    qgrams[l] = qgrams[j];
    qgrams[j] = temp1;

    // swap(&arr2[l], &arr2[j]);
    unsigned int temp2 = index[l];
    index[l] = index[j];
    index[j] = temp2;

    return j;
}

void quicksort_aux_3 (
    char** qgrams, unsigned int* index,
    int l, int r
)
{
    int j;

    if ( l < r )
    {
        // divide and conquer
        j = partition_3(qgrams, index, l, r);
        quicksort_aux_3(qgrams, index, l, j-1);
        quicksort_aux_3(qgrams, index, j+1, r);
    }
}

void quicksort_3 (
    char** qgrams, unsigned int* index, unsigned int n
)
{
    quicksort_aux_3(qgrams, index, 0, n-1);
}

// This one is meant to be called...
void sort_qgrams_lexicographically (
    char** qgrams, unsigned int* index, unsigned int n
)
{
    printf("sort_qgrams_lexicographically\n");
    quicksort_3(qgrams,index,n);
}
