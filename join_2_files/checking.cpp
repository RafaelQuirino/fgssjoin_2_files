#include "checking.h"
#include "similarity.h"
#include "compaction.h"
#include "util.h"

void checking_kernel (
	unsigned int* buckets, unsigned short* scores,
	vector< vector<token_t> > tsets,
	vector< vector<token_t> > tsets_2,
	float threshold,
    short* partial_scores,
	unsigned int csize,
	int num_columns
)
{
    unsigned int i;
    //int n = (int) tsets.size();

    for (i = 0; i < csize; i++)
	{
		unsigned int bucket = buckets[i];

		unsigned int query  = bucket / num_columns;
		unsigned int source = bucket % num_columns;
		unsigned int query_len  = tsets[query].size();
		unsigned int source_len = tsets_2[source].size();
		float minoverlap = h_jac_min_overlap(query_len, source_len, threshold);


		// Simpler version, without reusing
		// partial scores information...
		//----------------------------------
		unsigned int p1 = 0, p2 = 0;
		unsigned short score = 0;;


		while (p1 < query_len && p2 < source_len)
		{
            unsigned int tkn1 = tsets[query][p1].order_1_id;
            unsigned int tkn2 = tsets_2[source][p2].order_1_id;

			if ((p1 == query_len-1 && tkn1 < tkn2) ||
				(p2 == source_len-1 && tkn2 < tkn1))
				break;

			if (tkn1 == tkn2)
			{
				score++;
				p1++; p2++;
			}
			else
			{
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



double checking (
	vector< vector<token_t> > tsets,
	vector< vector<token_t> > tsets_2,
	unsigned int* buckets,
	short* partial_scores, float threshold, unsigned int csize,
	unsigned int** similar_pairs, unsigned short** scores_out, int* num_pairs
)
{
	unsigned long t0, t1, t00, t01;
	double mem;

    fprintf(stderr, "=> CHECKING...\n");
    t00 = getTimeInMicroseconds();

	mem = (double)(csize*sizeof(unsigned short))/1024.0;
	fprintf(stderr, "\t. Allocating %g KB on cpu (scores).\n", mem);
	unsigned short* scores;
	scores = (unsigned short*) malloc(csize * sizeof(unsigned short));
	memset(scores, (unsigned short) 0, csize * sizeof(unsigned short));

	fprintf(stderr, "\t* Calling checking_kernel...\n");
	t0 = getTimeInMicroseconds();

	checking_kernel(
		buckets, scores,
		tsets, tsets_2, threshold,
		partial_scores, csize,
		tsets_2.size()
	);

	t1 = getTimeInMicroseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", intervalInMiliseconds(t0,t1));

    // COMPACTING SIMILAR PAIRS ------------------------------------------------
	fprintf(stderr, "\t* Compacting similar pairs...\n");
	t0 = getTimeInMicroseconds();

    int nres = 0;

    filter_k_2 (scores,scores,buckets,buckets, &nres, csize);
    unsigned int comp_pairs_size = (unsigned int) nres;
    buckets = (unsigned int*) realloc(buckets, nres * sizeof(unsigned int));
    scores = (unsigned short*) realloc(scores, nres * sizeof(unsigned short));

	t1 = getTimeInMicroseconds();
	fprintf(stderr, "\t> Done. It took %g ms.\n", intervalInMiliseconds(t0,t1));
    //--------------------------------------------------------------------------

	fprintf(stderr, "\t# NUMBER OF SIMILAR PAIRS: %u\n", comp_pairs_size);

	t01 = getTimeInMicroseconds();
	fprintf(stderr, "DONE IN %g ms.\n\n", intervalInMiliseconds(t00,t01));

	//---------------------------
    *scores_out = scores;
	*similar_pairs = buckets;
    *num_pairs = comp_pairs_size;
	//---------------------------

	return intervalInMiliseconds(t00,t01);
}



void checking_kernel_2 (
	vector< vector<token_t> > tsets_orig, unsigned int* doc_index,
	unsigned int* buckets, unsigned short* scores,
	vector< vector<token_t> > tsets, float threshold,
    short* partial_scores,
	unsigned int csize
)
{
    int n = tsets.size();

    for (unsigned i = 0; i < csize; i++)
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
		unsigned short score = 0;


		while (p1 < query_len && p2 < source_len)
		{
            unsigned int tkn1 = tsets[query][p1].order_2_id;
            unsigned int tkn2 = tsets[source][p2].order_2_id;

			if ((p1 == query_len-1 && tkn1 < tkn2) ||
				(p2 == source_len-1 && tkn2 < tkn1))
				break;

			if (tkn1 == tkn2)
			{
				score++;
				p1++; p2++;
			}
			else
			{
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



double checking_2 (
	vector< vector<token_t> > tsets_orig, unsigned int* doc_index,
	vector< vector<token_t> > tsets, unsigned int* buckets,
	short* partial_scores, float threshold, unsigned int csize,
	unsigned int** similar_pairs, unsigned short** scores_out, int* num_pairs
)
{
	unsigned long t0, t1, t00, t01;

    cout << "=> CHECKING...\n";
    t00 = getTimeInMicroseconds();

	cout << "\t. Allocating " << (double)(csize*sizeof(unsigned short))/1024.0;
	cout << " KB on cpu (scores).\n";
	unsigned short* scores;
	scores = (unsigned short*) malloc(csize * sizeof(unsigned short));
	memset(scores, (unsigned short) 0, csize * sizeof(unsigned short));

	cout << "\t* Calling checking_kernel...\n";
	t0 = getTimeInMicroseconds();

	checking_kernel_2(
		tsets_orig, doc_index,
		buckets, scores,
		tsets, threshold,
		partial_scores, csize
	);

	t1 = getTimeInMicroseconds();
	cout << "\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";

    // COMPACTING SIMILAR PAIRS ------------------------------------------------
	cout << "\t* Compacting similar pairs...\n";
	t0 = getTimeInMicroseconds();

    int nres = 0;

    filter_k_2 (scores,scores,buckets,buckets, &nres, csize);
    unsigned int comp_pairs_size = (unsigned int) nres;
    buckets = (unsigned int*) realloc(buckets, nres * sizeof(unsigned int));
    scores = (unsigned short*) realloc(scores, nres * sizeof(unsigned short));

	t1 = getTimeInMicroseconds();
	cout << "\t> Done. It took " << intervalInMiliseconds(t0,t1) << " ms.\n";
    //--------------------------------------------------------------------------

	cout << "\t# NUMBER OF SIMILAR PAIRS: " << comp_pairs_size << "\n";

	t01 = getTimeInMicroseconds();
	cout << "DONE IN " << intervalInMiliseconds(t00,t01) << " ms.\n\n";

	//---------------------------
    *scores_out = scores;
	*similar_pairs = buckets;
    *num_pairs = comp_pairs_size;
	//---------------------------

	return intervalInMiliseconds(t00,t01);
}



void print_similar_pairs (
	vector<string> data,
	vector< vector<token_t> > tsets, unsigned int* doc_index,
	vector< vector<token_t> > tsets_2, unsigned int* doc_index_2,
	unsigned int* similar_pairs, unsigned short* scores, int num_pairs
)
{
	for (int i = 0; i < num_pairs; i++)
	{
		unsigned long n     = tsets_2.size();
		unsigned int  pair  = similar_pairs[i];
		unsigned int query  = pair / n;
		unsigned int source = pair % n;

		float sim = h_jac_similarity(
			tsets[query].size(), tsets_2[source].size(), scores[i]
		);

		printf("%u %u %f\n",
			doc_index[query]+1, doc_index_2[source]+1, sim
		);

		if (doc_index[query]+1 == 73)
			// tk_print_tset(tsets[query], QGRAM);
			printf("[%s]\n", data[query].c_str());
	}
}
