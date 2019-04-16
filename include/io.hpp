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

#include "util.hpp"



using namespace std;



/* DOCUMENTATION
 *
 */
vector<string> io_get_input_from_file (string filename);



/* DOCUMENTATION
 *
 */
vector<string> io_get_partial_input_from_file (string filename, int nlines);



/* DOCUMENTATION
 *
 */
void io_read_file_backup (string &filename, vector<string> &inputs);



/* DOCUMENTATION
 *
 */
void io_write_output_file (
	unsigned int* pos, unsigned int* len, unsigned int* tokens,
	unsigned int n_sets, unsigned int n_tokens, unsigned int n_terms,
	int input_size, int qgram, string path, const char* outdir
);



#endif // _IO_H_

