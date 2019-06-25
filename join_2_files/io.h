#ifndef _IO_H_
#define _IO_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "util.h"

using namespace std;

vector<string> getInputFromFile (string filename);

vector<string> getPartialInputFromFile (string filename, int nlines);

void readFileBackup (string &filename, vector<string> &inputs);

void writeOutputFile (unsigned int* pos, unsigned int* len, unsigned int* tokens,
	unsigned int n_sets, unsigned int n_tokens, unsigned int n_terms,
	int input_size, int qgram, string path, const char* outdir);

#endif
