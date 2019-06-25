#ifndef _DATA_H_
#define _DATA_H_

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using namespace std;

vector<string>
get_input_data (string file_path, int input_size);

vector<string>
get_proc_data (vector<string> input_data);

void proc_data (vector<string>& inpud_data);

#endif
