#ifndef _QGRAM_H_
#define _QGRAM_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "token.h"

using namespace std;

struct _qgram {

};

typedef struct _qgram qgram_t;

vector<string> qg_get_record (string str, int qgramsize);
vector< vector<string> > qg_get_records (vector<string> data, int qgramsize);

vector< vector<unsigned long> >
qg_get_sets (vector< vector<string> > recs);

void qg_print_record  (vector<string> rec);
void qg_print_records (vector< vector<string> > recs);
void qg_print_set  (vector<unsigned long> set);
void qg_print_sets (vector< vector<unsigned long> > sets);

unsigned long qg_hash (string qgram);

unordered_map<unsigned long,token_t>
qg_get_dict (
    vector< vector<string> > recs
);

unordered_map<unsigned long,token_t>
qg_get_dict_2 (
    vector< vector<string> > recs,
    vector< vector<string> > recs_2
);

void qg_print_dict (unordered_map<unsigned long,token_t> dict);

#endif // _QGRAM_H_
