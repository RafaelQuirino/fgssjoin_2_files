#include "similarity.hpp"
#include "checking.hpp"
#include "compaction.h"
#include "util.hpp"



/* DOCUMENTATION
 *
 */
void checking_kernel (
	unsigned int* buckets, unsigned short* scores,
	vector< vector<token_t> > tsets, float threshold,
    short* partial_scores,
	unsigned int csize
)
{
    unsigned int i;
    int n = tsets.size();

    for (i = 0; i < csize; i++)
	{
		unsigned int bucket = buckets[i];

		unsigned int query  = bucket / n;
		unsigned int source = bucket % n;
		unsigned int query_len  = tsets[query].size();
		unsigned int source_len = tsets[source].size();
		float minoverlap = h_jac_min_overlap(query_len, source_len, threshold);


		// Simpler version, without reusing
		// partial scores information...
		//----------------------------------
		unsigned int p1 = 0, p2 = 0;
		unsigned short score = 0;;


		while (p1 < query_len && p2 < source_len)
		{
            unsigned int tkn1 = tsets[query][p1].order_id;
            unsigned int tkn2 = tsets[source][p2].order_id;

			if ((p1 == query_len-1 && tkn1 < tkn2) ||
				(p2 == source_len-1 && tkn2 < tkn1)) {
				break;
			}

			if (tkn1 == tkn2) {
				score++;
				p1++; p2++;
			}
			else {
				// Sophisticated solution, with positional filtering
				//---------------------------------------------------
				unsigned int whichset = tkn1 < tkn2 ? 1 : 2;
				unsigned int rem;
				if (whichset == 1) rem = (query_len  - p1) - 0;
				else               rem = (source_len - p2) - 0;

				if ((rem + score) < minoverlap) {
					score = 0;
					break;
				}

				if (whichset == 1) p1++;
				else               p2++;

			} // if (tkn1 == tkn2)
		} // while (p1 < query_len && p2 < source_len)


		// Just testing minimum overlap score
		//------------------------------------
		float fscore = 0.0f;
		fscore += score;
		if (fscore >= minoverlap)
			scores[i] = score;
		else
			scores[i] = 0;

	} // for (i = 0; i < csize; i++)
} // void checking_kernel



/* DOCUMENTATION
 *
 */
double checking (
	vector< vector<token_t> > tsets, unsigned int* buckets,
	short* partial_scores, float threshold, unsigned int csize,
	unsigned int** similar_pairs, unsigned short** scores_out, int* num_pairs
)
{
	unsigned long t0, t1, t00, t01;

    cout << "=> CHECKING...\n";
    t00 = ut_get_time_in_microseconds();

	cout << "\t. Allocating " << (double)(csize * sizeof(unsigned short)) / 1024.0;
	cout << " KB on cpu (scores).\n";
	unsigned short* scores;
	scores = (unsigned short*) malloc (csize * sizeof(unsigned short));
	memset (scores, (unsigned short) 0, csize * sizeof(unsigned short));

	cout << "\t* Calling checking_kernel...\n";
	t0 = ut_get_time_in_microseconds();

	checking_kernel (
		buckets, scores,
		tsets, threshold,
		partial_scores, csize
	);

	t1 = ut_get_time_in_microseconds();
	cout << "\t> Done. It took " << ut_interval_in_miliseconds(t0,t1) << " ms.\n";

    // COMPACTING SIMILAR PAIRS ------------------------------------------------
	cout << "\t* Compacting similar pairs...\n";
	t0 = ut_get_time_in_microseconds();

    int nres = 0;

    filter_k_2 (scores, scores, buckets, buckets, &nres, csize);
    unsigned int comp_pairs_size = (unsigned int) nres;
    buckets = (unsigned int*) realloc (buckets, nres * sizeof(unsigned int));
    scores  = (unsigned short*) realloc (scores, nres * sizeof(unsigned short));

	t1 = ut_get_time_in_microseconds();
	cout << "\t> Done. It took " << ut_interval_in_miliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

	cout << "\t# NUMBER OF SIMILAR PAIRS: " << comp_pairs_size << "\n";

	t01 = ut_get_time_in_microseconds();
	cout << "DONE IN " << ut_interval_in_miliseconds (t00,t01) << " ms.\n\n";

	//---------------------------
    *scores_out = scores;
	*similar_pairs = buckets;
    *num_pairs = comp_pairs_size;
	//---------------------------

	return ut_interval_in_miliseconds (t00,t01);
}



/* DOCUMENTATION
 *
 */
void print_similar_pairs (
	vector< vector<token_t> > tsets, unsigned int* doc_index,
	unsigned int* similar_pairs, unsigned short* scores, int num_pairs
)
{
	for (int i = 0; i < num_pairs; i++)
	{
		unsigned long n     = tsets.size();
		unsigned int  pair  = similar_pairs[i];
		unsigned int query  = pair / n;
		unsigned int source = pair % n;

		float sim = h_jac_similarity(
			tsets[query].size(), tsets[source].size(), scores[i]
		);

		printf("%u %u %f\n",
			doc_index[query]+1, doc_index[source]+1, sim
		);
	}
}
