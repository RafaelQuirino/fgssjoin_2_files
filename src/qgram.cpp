#include "util.hpp"
#include "qgram.hpp"



/* DOCUMENTATION
 *
 */
vector<string> qg_get_record (string str, int qgramsize)
{
	vector<string> record;
    int nqgrams = str.size()-qgramsize+1;

	unordered_map<string,int> map;

	for (int i = 0; i < nqgrams; i++)
	{
        int occ = 0; // Occurrence of a qgram a string (0th, 1st, 2nd, ...)
        string qgram = str.substr(i,qgramsize);
        unordered_map<string,int>::const_iterator result = map.find(qgram);
        
		if (result == map.end()) {
            map[qgram] = 0;
        }
        else {
            map[qgram] += 1;
            occ = map[qgram];
        }

        char numstr[21]; // Enough to hold all numbers up to 64-bits
        sprintf (numstr, "%d", occ);
        string newqgram = qgram + numstr;

        record.push_back(newqgram);
    }

	return record;
}



/* DOCUMENTATION
 *
 */
vector< vector<string> > qg_get_records (vector<string> data, int qgramsize)
{
	fprintf (stderr, "\tCreating %d-gram records...\n", qgramsize);
    unsigned long t0 = ut_get_time_in_microseconds();

	vector< vector<string> > records;
	for (int i = 0; i < data.size(); i++)
	{
		records.push_back( qg_get_record (data[i], qgramsize) );
	}

	unsigned long t1 = ut_get_time_in_microseconds();
    fprintf(stderr, "\t> Done in %gms.\n", ut_interval_in_miliseconds (t0,t1));

	return records;
}



/* DOCUMENTATION
 *
 */
vector< vector<unsigned long> >
qg_get_sets (vector< vector<string> > recs)
{
	vector< vector<unsigned long> > sets;

	for (int i = 0; i < recs.size(); i++)
	{
		vector<unsigned long> set;
		for (int j = 0; j < recs[i].size(); j++)
		{
			set.push_back(qg_hash(recs[i][j]));
		}

		sets.push_back(set);
	}

	return sets;
}



/* DOCUMENTATION
 *
 */
void qg_print_record (vector<string> rec)
{
	printf("[");
	for (int i = 0; i < rec.size(); i++)
	{
		if (i == rec.size()-1)
			printf("'%s']\n", rec[i].c_str());
		else
			printf("'%s', ", rec[i].c_str());
	}
}

void qg_print_records (vector< vector<string> > recs)
{
	for (int i = 0; i < recs.size(); i++)
	{
		qg_print_record(recs[i]);
		printf("\n");
	}
}

void qg_print_set (vector<unsigned long> set)
{
	printf("[");
	for (int i = 0; i < set.size(); i++)
	{
		if (i == set.size()-1)
			printf("%lu]\n", set[i]);
		else
			printf("%lu, ", set[i]);
	}
}

void qg_print_sets (vector< vector<unsigned long> > sets)
{
	for (int i = 0; i < sets.size(); i++)
	{
		qg_print_set(sets[i]);
		printf("\n");
	}
}



/* DOCUMENTATION
 *
 */
unsigned long qg_hash (string qgram)
{
	char* qg = (char*) qgram.c_str();
	return ut_sdbm_hash(qg);
}



/* DOCUMENTATION
 *
 */
unordered_map<unsigned long,token_t>
qg_get_dict (vector< vector<string> > records)
{
	unordered_map<unsigned long,token_t> dict;

	for (int i = 0; i < records.size(); i++)
	{
		for (int j = 0; j < records[i].size(); j++)
		{
			string qgram = records[i][j];
			unsigned long hcode = qg_hash(qgram);

			// First check if token is already in dictionary.
            // If not, create it. If it is, increment frequency.
            unordered_map<unsigned long,token_t>::const_iterator result;
			result = dict.find(hcode);

			// If entry not found in dictionary
            if (result == dict.end()) {
                token_t newtkn;
				newtkn.qgram = qgram;
				newtkn.hash = hcode;
                newtkn.freq = 1;
                newtkn.doc_id   = -1;
                newtkn.order_id = -1;
                dict[hcode] = newtkn;
            }
            else {
                dict[hcode].freq += 1;
            }
		}
	}

	return dict;
}



/* DOCUMENTATION
 *
 */
void qg_print_dict (unordered_map<unsigned long,token_t> dict)
{
	for (pair<unsigned long,token_t> element : dict)
    {
		cout << "(" << element.first << ") :: " << endl;
		cout << " \t qgram: (" << element.second.qgram << ")" << endl;
		cout << " \t hash: " << element.second.hash << endl;
		cout << " \t freq: " << element.second.freq << endl;
		cout << " \t doc_id: " << element.second.doc_id << endl;
		cout << " \t order_id: " << element.second.order_id << endl;
		cout << endl;
    }
}
