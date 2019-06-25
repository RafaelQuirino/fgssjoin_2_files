#include "io.h"

vector<string> getInputFromFile (string filename)
{
    ifstream input(filename.c_str());
    string line;

    vector<string> inputs;

    if (input.is_open()) {
	    while (!input.eof()) {
	        getline(input, line);
            // if(line == "" || line.size() < 2) continue;
            if(line.size() < 2) continue;
			inputs.push_back(line);
	    }
    }

    input.close();

    return inputs;
}

vector<string> getPartialInputFromFile (string filename, int n)
{
    ifstream input(filename.c_str());
    string line;

    vector<string> inputs;
    int i = 0;
    if (input.is_open()) {
        while (!input.eof() && i++ < n) {
            getline(input, line);
            if(line == "") continue;
            inputs.push_back(line);

            if (i >= n) break;
        }
    }

    input.close();

    return inputs;
}

void readFileBackup (string &filename, vector<string>& inputs)
{
    ifstream input(filename.c_str());
    string line;

    if (input.is_open()) {
	    while (!input.eof()) {
	        getline(input, line);
	        if(line == "") continue;
			inputs.push_back(line);
	    }
    }

    input.close();
}

void writeOutputFile (unsigned int* pos, unsigned int* len, unsigned int* tokens,
    unsigned int n_sets, unsigned int n_tokens, unsigned int n_terms,
    int input_size, int qgram, string path, const char* outdir)
{
    char fname[128];
    vector<string> v = split(path, '/');
    const char* oldname = split(v[v.size()-1], '.')[0].c_str();
    if (input_size == 0 || input_size >= (int) n_sets)
        input_size = n_sets;

    char* slash = outdir[strlen(outdir)-1] == '/' ? (char*)"" : (char*)"/";
    sprintf(fname, "%s%s%s_%d_%d.sets", outdir, slash, oldname, input_size, qgram);
    //sprintf(fname, "%s_%d_%d.sets", oldname, input_size, qgram);

    printf("Writing file: %s\n", fname);

    FILE *fp;
    fp = fopen(fname, "wb");

    unsigned int sizes[3] = {n_sets, n_tokens, n_terms};
    fwrite(sizes, sizeof(unsigned int), 3, fp);
    fwrite(pos, sizeof(unsigned int), n_sets, fp);
    fwrite(len, sizeof(unsigned int), n_sets, fp);
    fwrite(tokens, sizeof(unsigned int), n_tokens, fp);

    fclose(fp);
}
